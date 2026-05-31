// SPDX-License-Identifier: GPL-2.0
/*
 * KV260 NPU driver for Ubuntu/XMUtil dynamic PL loading.
 *
 * The zcu102 driver mapped a fixed reserved-memory DDR window. This driver
 * allocates coherent DMA memory from Linux CMA and exposes the DMA address to
 * user space for the NPU's 32-bit AXI master.
 */
#include <linux/bitops.h>
#include <linux/cdev.h>
#include <linux/clk.h>
#include <linux/completion.h>
#include <linux/delay.h>
#include <linux/dmaengine.h>
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/overflow.h>
#include <linux/platform_device.h>
#include <linux/reset.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/wait.h>

#include "npu_kv260_uapi.h"

#define DRIVER_NAME "npu_kv260"
#define CLASS_NAME  "npu_kv260_class"

#define REG_DECODE_IAR 0xB8
#define REG_DECODE_MER 0xC0
#define REG_DECODE_IER 0xC8
#define REG_DECODE_ISR 0xD0
#define REG_DECODE_IPR 0xD8

#define REG_PREFILL_IAR 0x100
#define REG_PREFILL_MER 0x108
#define REG_PREFILL_IER 0x110
#define REG_PREFILL_ISR 0x118
#define REG_PREFILL_IPR 0x120

#define REG_GENERIC_MAGIC   0xFF00
#define REG_GENERIC_VERSION 0xFF08
#define REG_GENERIC_MODE    0xFF10
#define REG_GENERIC_CAPS    0xFF18
#define REG_GENERIC_CONTROL 0xFF20
#define REG_GENERIC_STATUS  0xFF28
#define REG_GENERIC_ERROR   0xFF30

#define GENERIC_CONTROL_SOFT_RESET  BIT(0)
#define GENERIC_CONTROL_IRQ_CLEAR   BIT(1)
#define GENERIC_CONTROL_ERROR_CLEAR BIT(2)

#define DEFAULT_REG_SIZE 0x10000UL

static unsigned long reg_base;
module_param(reg_base, ulong, 0444);
MODULE_PARM_DESC(reg_base, "Fallback MMIO base address when no DT platform device is present");

static unsigned int reg_size = DEFAULT_REG_SIZE;
module_param(reg_size, uint, 0444);
MODULE_PARM_DESC(reg_size, "Fallback MMIO register window size");

static int irq = -1;
module_param(irq, int, 0444);
MODULE_PARM_DESC(irq, "Fallback Linux IRQ number; leave unset for userspace polling only");

static unsigned int max_buffer_mb = 1024;
module_param(max_buffer_mb, uint, 0644);
MODULE_PARM_DESC(max_buffer_mb, "Maximum CMA buffer size accepted by the driver");

static unsigned int dma_copy_chunk_kb = 1024;
module_param(dma_copy_chunk_kb, uint, 0644);
MODULE_PARM_DESC(dma_copy_chunk_kb, "Maximum bounce-buffer chunk size for DMA copy ioctl");

static unsigned int dma_copy_timeout_ms = 5000;
module_param(dma_copy_timeout_ms, uint, 0644);
MODULE_PARM_DESC(dma_copy_timeout_ms, "Timeout for each DMA copy chunk");

static atomic_t active_devices = ATOMIC_INIT(0);

struct npu_dev {
    struct device *dev;
    void __iomem *base_addr;
    phys_addr_t regs_phys_base;
    resource_size_t regs_size;
    int irq;
    bool has_irq;

    struct clk *npu_clk;
    struct clk *aclk;
    struct clk *prefill_clk;
    struct clk *decode_clk;
    struct reset_control *rst;

    dev_t dev_num;
    struct cdev cdev;
    struct class *class;
    struct device *char_dev;

    wait_queue_head_t wait_q;
    atomic_t irq_done;
    atomic_t irq_mode;
    u32 last_isr_status;

    struct mutex state_lock;
    u32 hw_magic;
    u32 hw_abi_version;
    u32 hw_mode_id;
    u32 hw_caps;
    u32 hw_status;
    u32 hw_error;
    u32 reg_iar;
    u32 reg_mer;
    u32 reg_ier;
    u32 reg_isr;
    u32 reg_ipr;
    bool hw_ready;

    struct dma_chan *dma_chan;
    struct mutex dma_lock;
};

struct npu_file_ctx {
    struct npu_dev *npu;
    struct mutex lock;
    void *cpu_addr;
    struct device *dma_dev;
    dma_addr_t dma_addr;
    size_t size;
};

static struct platform_device *fallback_pdev;

struct npu_shared_buffer {
    struct mutex lock;
    struct device *dev;
    void *cpu_addr;
    dma_addr_t dma_addr;
    size_t size;
};

static struct npu_shared_buffer shared_buffer = {
    .lock = __MUTEX_INITIALIZER(shared_buffer.lock),
};

static void npu_write(struct npu_dev *npu, u32 offset, u32 val)
{
    iowrite32(val, npu->base_addr + offset);
}

static u32 npu_read(struct npu_dev *npu, u32 offset)
{
    return ioread32(npu->base_addr + offset);
}

static int npu_reg_bounds_check(struct npu_dev *npu, u32 offset, size_t width)
{
    if (offset + width < offset)
        return -EINVAL;
    if (offset + width > npu->regs_size)
        return -EINVAL;
    return 0;
}

static int npu_hw_reset(struct npu_dev *npu)
{
    int ret;

    if (!npu->rst)
        return 0;

    ret = reset_control_assert(npu->rst);
    if (ret)
        return ret;

    usleep_range(10, 20);

    ret = reset_control_deassert(npu->rst);
    if (ret)
        return ret;

    return 0;
}

static struct clk *npu_runtime_clk(struct npu_dev *npu)
{
    if (npu->hw_mode_id == NPU_KV260_MODE_PREFILL && npu->prefill_clk)
        return npu->prefill_clk;
    if (npu->hw_mode_id == NPU_KV260_MODE_DECODE && npu->decode_clk)
        return npu->decode_clk;
    if (npu->npu_clk)
        return npu->npu_clk;
    if (npu->aclk)
        return npu->aclk;
    if (npu->prefill_clk)
        return npu->prefill_clk;
    return npu->decode_clk;
}

static int npu_set_runtime_clock_rate(struct npu_dev *npu,
                                      struct npu_kv260_clock_rate *rate)
{
    struct clk *clk = npu_runtime_clk(npu);
    int ret;

    if (!clk)
        return -EOPNOTSUPP;
    if (!rate->requested_hz || rate->requested_hz > ULONG_MAX)
        return -EINVAL;

    ret = clk_set_rate(clk, (unsigned long)rate->requested_hz);
    if (ret)
        return ret;

    rate->actual_hz = clk_get_rate(clk);
    return 0;
}

static void npu_apply_irq_mode(struct npu_dev *npu);

static void npu_use_decode_irq_regs(struct npu_dev *npu)
{
    npu->reg_iar = REG_DECODE_IAR;
    npu->reg_mer = REG_DECODE_MER;
    npu->reg_ier = REG_DECODE_IER;
    npu->reg_isr = REG_DECODE_ISR;
    npu->reg_ipr = REG_DECODE_IPR;
}

static void npu_use_prefill_irq_regs(struct npu_dev *npu)
{
    npu->reg_iar = REG_PREFILL_IAR;
    npu->reg_mer = REG_PREFILL_MER;
    npu->reg_ier = REG_PREFILL_IER;
    npu->reg_isr = REG_PREFILL_ISR;
    npu->reg_ipr = REG_PREFILL_IPR;
}

static void npu_fill_hw_state(struct npu_dev *npu,
                              struct npu_kv260_hw_state *state)
{
    memset(state, 0, sizeof(*state));
    state->magic = npu->hw_magic;
    state->abi_version = npu->hw_abi_version;
    state->mode_id = npu->hw_mode_id;
    state->caps = npu->hw_caps;
    state->status = npu->hw_status;
    state->error = npu->hw_error;
}

static int npu_reinit_hw_locked(struct npu_dev *npu,
                                struct npu_kv260_hw_state *state)
{
    u32 magic;
    u32 mode;

    if (npu_reg_bounds_check(npu, REG_GENERIC_ERROR, sizeof(u32)))
        return -ERANGE;

    magic = npu_read(npu, REG_GENERIC_MAGIC);
    if (magic != NPU_KV260_HW_MAGIC) {
        npu->hw_ready = false;
        npu->hw_magic = magic;
        npu->hw_abi_version = 0;
        npu->hw_mode_id = NPU_KV260_MODE_UNKNOWN;
        npu->hw_caps = 0;
        npu->hw_status = 0;
        npu->hw_error = 0;
        return -ENODEV;
    }

    mode = npu_read(npu, REG_GENERIC_MODE);
    if (mode == NPU_KV260_MODE_PREFILL) {
        npu_use_prefill_irq_regs(npu);
    } else if (mode == NPU_KV260_MODE_DECODE) {
        npu_use_decode_irq_regs(npu);
    } else {
        npu->hw_ready = false;
        return -EINVAL;
    }

    npu->hw_magic = magic;
    npu->hw_abi_version = npu_read(npu, REG_GENERIC_VERSION);
    npu->hw_mode_id = mode;
    npu->hw_caps = npu_read(npu, REG_GENERIC_CAPS);
    npu->hw_status = npu_read(npu, REG_GENERIC_STATUS);
    npu->hw_error = npu_read(npu, REG_GENERIC_ERROR);
    npu->hw_ready = true;

    npu_write(npu, REG_GENERIC_CONTROL,
              GENERIC_CONTROL_IRQ_CLEAR | GENERIC_CONTROL_ERROR_CLEAR);
    atomic_set(&npu->irq_done, 0);
    npu->last_isr_status = 0;
    npu_apply_irq_mode(npu);

    if (state)
        npu_fill_hw_state(npu, state);
    return 0;
}

static void npu_apply_irq_mode(struct npu_dev *npu)
{
    if (npu_reg_bounds_check(npu, npu->reg_mer, sizeof(u32)) ||
        npu_reg_bounds_check(npu, npu->reg_ier, sizeof(u32)))
        return;

    npu_write(npu, npu->reg_mer, 0x03);

    if (atomic_read(&npu->irq_mode) == NPU_KV260_IRQ_MODE_KERNEL &&
        npu->has_irq) {
        npu_write(npu, npu->reg_ier, 0x3F);
    } else {
        npu_write(npu, npu->reg_ier, 0x00);
    }
}

static irqreturn_t npu_irq_handler(int irq_num, void *data)
{
    struct npu_dev *npu = data;
    u32 status;

    if (atomic_read(&npu->irq_mode) == NPU_KV260_IRQ_MODE_USERSPACE)
        return IRQ_HANDLED;

    if (npu_reg_bounds_check(npu, npu->reg_ipr, sizeof(u32)) ||
        npu_reg_bounds_check(npu, npu->reg_iar, sizeof(u32)))
        return IRQ_NONE;

    status = npu_read(npu, npu->reg_ipr);
    if (!status)
        return IRQ_NONE;

    npu_write(npu, npu->reg_iar, status);
    npu->last_isr_status = status;
    atomic_set(&npu->irq_done, 1);
    wake_up_interruptible(&npu->wait_q);
    return IRQ_HANDLED;
}

static void npu_free_ctx_buffer(struct npu_file_ctx *ctx)
{
    if (!ctx->cpu_addr)
        return;

    /*
     * Keep one module-global coherent buffer so multiple userspace runtimes
     * can reuse the same CMA window across overlay switches.
     */
    ctx->cpu_addr = NULL;
    ctx->dma_dev = NULL;
    ctx->dma_addr = 0;
    ctx->size = 0;
}

static int npu_open(struct inode *inode, struct file *file)
{
    struct npu_dev *npu = container_of(inode->i_cdev, struct npu_dev, cdev);
    struct npu_file_ctx *ctx;

    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return -ENOMEM;

    ctx->npu = npu;
    mutex_init(&ctx->lock);
    file->private_data = ctx;

    npu_apply_irq_mode(npu);
    return 0;
}

static int npu_release(struct inode *inode, struct file *file)
{
    struct npu_file_ctx *ctx = file->private_data;

    if (!ctx)
        return 0;

    mutex_lock(&ctx->lock);
    npu_free_ctx_buffer(ctx);
    mutex_unlock(&ctx->lock);
    kfree(ctx);
    file->private_data = NULL;
    return 0;
}

static int npu_alloc_buffer(struct npu_file_ctx *ctx,
                            struct npu_kv260_buffer_request *req)
{
    size_t size;
    u64 max_size;
    int ret = 0;

    if (!req->size)
        return -EINVAL;

    if (ctx->cpu_addr)
        return -EBUSY;

    max_size = (u64)max_buffer_mb << 20;
    if (max_size && req->size > max_size)
        return -EINVAL;

    if (req->size > U32_MAX)
        return -EINVAL;

    size = PAGE_ALIGN((size_t)req->size);
    if (!size)
        return -EINVAL;

    mutex_lock(&shared_buffer.lock);
    if (shared_buffer.cpu_addr) {
        if (shared_buffer.size < size) {
            ret = -EINVAL;
            goto out_unlock;
        }
    } else {
        shared_buffer.cpu_addr = dma_alloc_coherent(ctx->npu->dev, size,
                                                    &shared_buffer.dma_addr,
                                                    GFP_KERNEL);
        if (!shared_buffer.cpu_addr) {
            ret = -ENOMEM;
            goto out_unlock;
        }
        shared_buffer.dev = get_device(ctx->npu->dev);
        shared_buffer.size = size;
    }

    if ((u64)shared_buffer.dma_addr > U32_MAX ||
        (u64)shared_buffer.dma_addr + shared_buffer.size - 1 > U32_MAX) {
        dev_err(ctx->npu->dev,
                "CMA buffer DMA address 0x%llx size 0x%zx exceeds NPU 32-bit address range\n",
                (unsigned long long)shared_buffer.dma_addr,
                shared_buffer.size);
        ret = -ERANGE;
        goto out_unlock;
    }

    ctx->cpu_addr = shared_buffer.cpu_addr;
    ctx->dma_dev = shared_buffer.dev;
    ctx->dma_addr = shared_buffer.dma_addr;
    ctx->size = shared_buffer.size;
    req->size = shared_buffer.size;
    req->dma_addr = ctx->dma_addr;

out_unlock:
    mutex_unlock(&shared_buffer.lock);
    return ret;
}

static void npu_dma_complete(void *arg)
{
    complete(arg);
}

static int npu_dmaengine_memcpy(struct npu_dev *npu, dma_addr_t dst,
                                dma_addr_t src, size_t bytes)
{
    struct dma_async_tx_descriptor *desc;
    DECLARE_COMPLETION_ONSTACK(done);
    dma_cookie_t cookie;
    enum dma_status status;
    unsigned long timeout;

    if (!npu->dma_chan)
        return -EOPNOTSUPP;

    desc = dmaengine_prep_dma_memcpy(npu->dma_chan, dst, src, bytes,
                                     DMA_CTRL_ACK | DMA_PREP_INTERRUPT);
    if (!desc)
        return -EIO;

    desc->callback = npu_dma_complete;
    desc->callback_param = &done;

    cookie = dmaengine_submit(desc);
    if (dma_submit_error(cookie))
        return dma_submit_error(cookie);

    dma_async_issue_pending(npu->dma_chan);
    timeout = msecs_to_jiffies(dma_copy_timeout_ms);
    if (!wait_for_completion_timeout(&done, timeout ? timeout : 1)) {
        dmaengine_terminate_sync(npu->dma_chan);
        return -ETIMEDOUT;
    }

    status = dma_async_is_tx_complete(npu->dma_chan, cookie, NULL, NULL);
    if (status != DMA_COMPLETE)
        return -EIO;

    return 0;
}

static int npu_dma_copy_ioctl(struct npu_file_ctx *ctx,
                              struct npu_kv260_dma_copy *req)
{
    struct npu_dev *npu = ctx->npu;
    struct device *dma_dev;
    void *bounce;
    u64 end;
    size_t chunk;
    size_t done = 0;
    ktime_t start;
    int ret = 0;

    if (!ctx->cpu_addr)
        return -ENODATA;
    if (!req->size)
        return -EINVAL;
    if (req->direction != NPU_KV260_DMA_COPY_CMA_TO_USER &&
        req->direction != NPU_KV260_DMA_COPY_USER_TO_CMA)
        return -EINVAL;
    if (req->cma_offset > ctx->size)
        return -EINVAL;
    if (check_add_overflow(req->cma_offset, req->size, &end) ||
        end > ctx->size)
        return -EINVAL;

    chunk = req->chunk_bytes ? req->chunk_bytes :
            (size_t)dma_copy_chunk_kb * 1024u;
    if (!chunk)
        chunk = 1024u * 1024u;
    if (chunk > 4u * 1024u * 1024u)
        chunk = 4u * 1024u * 1024u;
    chunk = PAGE_ALIGN(chunk);
    if (chunk > req->size)
        chunk = PAGE_ALIGN((size_t)req->size);

    if (!npu->dma_chan)
        return -EOPNOTSUPP;

    dma_dev = npu->dma_chan->device->dev;
    bounce = kmalloc(chunk, GFP_KERNEL);
    if (!bounce)
        return -ENOMEM;

    mutex_lock(&npu->dma_lock);
    start = ktime_get();
    while (done < req->size) {
        size_t todo = min_t(size_t, chunk, (size_t)(req->size - done));
        u64 user_addr = req->user_addr + done;
        dma_addr_t bounce_dma;
        enum dma_data_direction map_dir;

        if (req->direction == NPU_KV260_DMA_COPY_USER_TO_CMA) {
            if (copy_from_user(bounce, (void __user *)(unsigned long)user_addr,
                               todo)) {
                ret = -EFAULT;
                break;
            }
            map_dir = DMA_TO_DEVICE;
            bounce_dma = dma_map_single(dma_dev, bounce, todo, map_dir);
            if (dma_mapping_error(dma_dev, bounce_dma)) {
                ret = -EIO;
                break;
            }
            ret = npu_dmaengine_memcpy(npu, ctx->dma_addr + req->cma_offset + done,
                                       bounce_dma, todo);
            dma_unmap_single(dma_dev, bounce_dma, todo, map_dir);
        } else {
            map_dir = DMA_FROM_DEVICE;
            bounce_dma = dma_map_single(dma_dev, bounce, todo, map_dir);
            if (dma_mapping_error(dma_dev, bounce_dma)) {
                ret = -EIO;
                break;
            }
            ret = npu_dmaengine_memcpy(npu, bounce_dma,
                                       ctx->dma_addr + req->cma_offset + done,
                                       todo);
            dma_unmap_single(dma_dev, bounce_dma, todo, map_dir);
            if (!ret &&
                copy_to_user((void __user *)(unsigned long)user_addr, bounce,
                             todo)) {
                ret = -EFAULT;
            }
        }

        if (ret)
            break;
        done += todo;
    }
    req->elapsed_ns = ktime_to_ns(ktime_sub(ktime_get(), start));
    mutex_unlock(&npu->dma_lock);

    kfree(bounce);
    return ret;
}

static long npu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct npu_file_ctx *ctx = file->private_data;
    struct npu_dev *npu = ctx->npu;
    int ret = 0;

    switch (cmd) {
    case NPU_KV260_IOC_WAIT_IRQ: {
        u32 status;

        if (!npu->has_irq)
            return -EOPNOTSUPP;

        ret = wait_event_interruptible_timeout(
            npu->wait_q, atomic_read(&npu->irq_done) != 0, 2 * HZ);
        if (ret == 0)
            return -ETIMEDOUT;
        if (ret < 0)
            return -ERESTARTSYS;

        atomic_set(&npu->irq_done, 0);
        status = npu->last_isr_status;
        if (copy_to_user((u32 __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_GET_BUFFER_INFO: {
        struct npu_kv260_buffer_info info = {
            .size = ctx->size,
            .dma_addr = ctx->dma_addr,
        };

        if (!ctx->cpu_addr)
            return -ENODATA;
        if (copy_to_user((void __user *)arg, &info, sizeof(info)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_RESET_DEV:
        ret = npu_hw_reset(npu);
        if (ret)
            return ret;
        mutex_lock(&npu->state_lock);
        ret = npu_reinit_hw_locked(npu, NULL);
        mutex_unlock(&npu->state_lock);
        if (ret)
            return ret;
        npu_apply_irq_mode(npu);
        return 0;

    case NPU_KV260_IOC_SET_IRQ_MODE:
        if (arg > NPU_KV260_IRQ_MODE_USERSPACE)
            return -EINVAL;
        atomic_set(&npu->irq_mode, arg);
        if (arg == NPU_KV260_IRQ_MODE_KERNEL)
            atomic_set(&npu->irq_done, 0);
        npu_apply_irq_mode(npu);
        return 0;

    case NPU_KV260_IOC_ALLOC_BUFFER: {
        struct npu_kv260_buffer_request req;

        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;

        mutex_lock(&ctx->lock);
        ret = npu_alloc_buffer(ctx, &req);
        mutex_unlock(&ctx->lock);
        if (ret)
            return ret;

        if (copy_to_user((void __user *)arg, &req, sizeof(req)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_FREE_BUFFER:
        mutex_lock(&ctx->lock);
        npu_free_ctx_buffer(ctx);
        mutex_unlock(&ctx->lock);
        return 0;

    case NPU_KV260_IOC_GET_INFO: {
        struct npu_kv260_info info = {
            .regs_phys = npu->regs_phys_base,
            .regs_size = npu->regs_size,
            .dma_addr_bits = 32,
            .has_irq = npu->has_irq ? 1 : 0,
            .flags = npu->dma_chan ? NPU_KV260_INFO_FLAG_DMA_COPY : 0,
        };

        if (copy_to_user((void __user *)arg, &info, sizeof(info)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_REG_READ: {
        struct npu_kv260_reg_access access;

        if (copy_from_user(&access, (void __user *)arg, sizeof(access)))
            return -EFAULT;
        ret = npu_reg_bounds_check(npu, access.offset, sizeof(u32));
        if (ret)
            return ret;
        access.value = npu_read(npu, access.offset);
        if (copy_to_user((void __user *)arg, &access, sizeof(access)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_REG_WRITE: {
        struct npu_kv260_reg_access access;

        if (copy_from_user(&access, (void __user *)arg, sizeof(access)))
            return -EFAULT;
        ret = npu_reg_bounds_check(npu, access.offset, sizeof(u32));
        if (ret)
            return ret;
        npu_write(npu, access.offset, access.value);
        return 0;
    }

    case NPU_KV260_IOC_DMA_COPY: {
        struct npu_kv260_dma_copy copy;

        if (copy_from_user(&copy, (void __user *)arg, sizeof(copy)))
            return -EFAULT;
        ret = npu_dma_copy_ioctl(ctx, &copy);
        if (copy_to_user((void __user *)arg, &copy, sizeof(copy)))
            return -EFAULT;
        return ret;
    }

    case NPU_KV260_IOC_REINIT: {
        struct npu_kv260_hw_state state;

        mutex_lock(&npu->state_lock);
        ret = npu_reinit_hw_locked(npu, &state);
        mutex_unlock(&npu->state_lock);
        if (ret)
            return ret;
        if (copy_to_user((void __user *)arg, &state, sizeof(state)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_GET_HW_STATE: {
        struct npu_kv260_hw_state state;

        mutex_lock(&npu->state_lock);
        npu->hw_status = npu->hw_ready ? npu_read(npu, REG_GENERIC_STATUS) : 0;
        npu->hw_error = npu->hw_ready ? npu_read(npu, REG_GENERIC_ERROR) : 0;
        npu_fill_hw_state(npu, &state);
        mutex_unlock(&npu->state_lock);
        if (copy_to_user((void __user *)arg, &state, sizeof(state)))
            return -EFAULT;
        return 0;
    }

    case NPU_KV260_IOC_SET_CLOCK_RATE: {
        struct npu_kv260_clock_rate rate;

        if (copy_from_user(&rate, (void __user *)arg, sizeof(rate)))
            return -EFAULT;
        ret = npu_set_runtime_clock_rate(npu, &rate);
        if (ret)
            return ret;
        if (copy_to_user((void __user *)arg, &rate, sizeof(rate)))
            return -EFAULT;
        return 0;
    }

    default:
        return -EINVAL;
    }
}

static int npu_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct npu_file_ctx *ctx = file->private_data;
    struct npu_dev *npu = ctx->npu;
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long offset = vma->vm_pgoff;
    unsigned long regs_offset_pfn = NPU_KV260_MMAP_REGS_OFFSET >> PAGE_SHIFT;
    unsigned long regs_map_size = PAGE_ALIGN(npu->regs_size);

    vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);

    if (offset == 0) {
        int ret;

        mutex_lock(&ctx->lock);
        if (!ctx->cpu_addr || size != ctx->size) {
            mutex_unlock(&ctx->lock);
            return -EINVAL;
        }

        ret = dma_mmap_coherent(ctx->dma_dev ? ctx->dma_dev : npu->dev,
                                vma, ctx->cpu_addr, ctx->dma_addr,
                                ctx->size);
        mutex_unlock(&ctx->lock);
        return ret;
    }

    if (offset != regs_offset_pfn)
        return -EINVAL;

    if (size > regs_map_size)
        return -EINVAL;

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    if (remap_pfn_range(vma, vma->vm_start,
                        npu->regs_phys_base >> PAGE_SHIFT, size,
                        vma->vm_page_prot))
        return -EAGAIN;

    return 0;
}

static const struct file_operations npu_fops = {
    .owner = THIS_MODULE,
    .open = npu_open,
    .release = npu_release,
    .unlocked_ioctl = npu_ioctl,
    .mmap = npu_mmap,
};

static int npu_register_chrdev(struct npu_dev *npu)
{
    int ret;

    ret = alloc_chrdev_region(&npu->dev_num, 0, 1, DRIVER_NAME);
    if (ret)
        return ret;

    cdev_init(&npu->cdev, &npu_fops);
    ret = cdev_add(&npu->cdev, npu->dev_num, 1);
    if (ret)
        goto err_chrdev;

    npu->class = class_create(CLASS_NAME);
    if (IS_ERR(npu->class)) {
        ret = PTR_ERR(npu->class);
        goto err_cdev;
    }

    npu->char_dev = device_create(npu->class, NULL, npu->dev_num, NULL,
                                  NPU_KV260_DEVICE_NAME);
    if (IS_ERR(npu->char_dev)) {
        ret = PTR_ERR(npu->char_dev);
        goto err_class;
    }

    return 0;

err_class:
    class_destroy(npu->class);
err_cdev:
    cdev_del(&npu->cdev);
err_chrdev:
    unregister_chrdev_region(npu->dev_num, 1);
    return ret;
}

static void npu_unregister_chrdev(struct npu_dev *npu)
{
    device_destroy(npu->class, npu->dev_num);
    class_destroy(npu->class);
    cdev_del(&npu->cdev);
    unregister_chrdev_region(npu->dev_num, 1);
}

static int npu_probe(struct platform_device *pdev)
{
    struct npu_dev *npu;
    struct resource *res;
    int ret;

    npu = devm_kzalloc(&pdev->dev, sizeof(*npu), GFP_KERNEL);
    if (!npu)
        return -ENOMEM;

    if (atomic_cmpxchg(&active_devices, 0, 1) != 0) {
        dev_err(&pdev->dev, "only one KV260 NPU device is supported\n");
        return -EBUSY;
    }

    npu->dev = &pdev->dev;
    mutex_init(&npu->state_lock);
    mutex_init(&npu->dma_lock);
    npu_use_decode_irq_regs(npu);
    platform_set_drvdata(pdev, npu);

    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if (!res) {
        ret = -ENODEV;
        goto err_active;
    }

    npu->regs_phys_base = res->start;
    npu->regs_size = resource_size(res);
    npu->base_addr = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(npu->base_addr)) {
        ret = PTR_ERR(npu->base_addr);
        goto err_active;
    }

    if (!pdev->dev.dma_mask)
        pdev->dev.dma_mask = &pdev->dev.coherent_dma_mask;

    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    if (ret) {
        dev_err(&pdev->dev, "failed to set 32-bit DMA mask: %d\n", ret);
        goto err_active;
    }

    npu->irq = platform_get_irq_optional(pdev, 0);
    if (npu->irq >= 0) {
        ret = devm_request_irq(&pdev->dev, npu->irq, npu_irq_handler, 0,
                               DRIVER_NAME, npu);
        if (ret)
            goto err_active;
        npu->has_irq = true;
    } else {
        npu->has_irq = false;
        dev_warn(&pdev->dev, "no IRQ resource; runtime will use userspace polling only\n");
    }

    npu->npu_clk = devm_clk_get_optional(&pdev->dev, "npu_clk");
    if (IS_ERR(npu->npu_clk)) {
        ret = PTR_ERR(npu->npu_clk);
        goto err_active;
    }
    if (npu->npu_clk) {
        ret = clk_prepare_enable(npu->npu_clk);
        if (ret)
            goto err_active;
    }

    npu->prefill_clk = devm_clk_get_optional(&pdev->dev, "prefill_clk");
    if (IS_ERR(npu->prefill_clk)) {
        ret = PTR_ERR(npu->prefill_clk);
        goto err_npu_clk;
    }
    if (npu->prefill_clk) {
        ret = clk_prepare_enable(npu->prefill_clk);
        if (ret)
            goto err_npu_clk;
    }

    npu->decode_clk = devm_clk_get_optional(&pdev->dev, "decode_clk");
    if (IS_ERR(npu->decode_clk)) {
        ret = PTR_ERR(npu->decode_clk);
        goto err_prefill_clk;
    }
    if (npu->decode_clk) {
        ret = clk_prepare_enable(npu->decode_clk);
        if (ret)
            goto err_prefill_clk;
    }

    npu->aclk = devm_clk_get_optional(&pdev->dev, "clk");
    if (IS_ERR(npu->aclk)) {
        ret = PTR_ERR(npu->aclk);
        goto err_decode_clk;
    }
    if (npu->aclk) {
        ret = clk_prepare_enable(npu->aclk);
        if (ret)
            goto err_decode_clk;
    }

    npu->rst = devm_reset_control_get_optional_exclusive(&pdev->dev, "rst");
    if (IS_ERR(npu->rst)) {
        ret = PTR_ERR(npu->rst);
        goto err_clk;
    }

    ret = npu_hw_reset(npu);
    if (ret)
        goto err_clk;

    init_waitqueue_head(&npu->wait_q);
    atomic_set(&npu->irq_done, 0);
    atomic_set(&npu->irq_mode, NPU_KV260_IRQ_MODE_KERNEL);
    mutex_lock(&npu->state_lock);
    ret = npu_reinit_hw_locked(npu, NULL);
    mutex_unlock(&npu->state_lock);
    if (ret)
        dev_warn(&pdev->dev,
                 "generic capability probe failed (%d); REINIT will fail until a generic bitstream is loaded\n",
                 ret);

    {
        dma_cap_mask_t mask;

        dma_cap_zero(mask);
        dma_cap_set(DMA_MEMCPY, mask);
        npu->dma_chan = dma_request_channel(mask, NULL, NULL);
        if (!npu->dma_chan)
            dev_warn(&pdev->dev, "no DMA_MEMCPY channel available; DMA copy ioctl disabled\n");
    }

    ret = npu_register_chrdev(npu);
    if (ret)
        goto err_dma;

    dev_info(&pdev->dev,
             "KV260 NPU driver loaded: regs=0x%llx size=0x%llx irq=%d cma_max=%uMiB dma_copy=%s\n",
             (unsigned long long)npu->regs_phys_base,
             (unsigned long long)npu->regs_size,
             npu->has_irq ? npu->irq : -1, max_buffer_mb,
             npu->dma_chan ? dev_name(npu->dma_chan->device->dev) : "disabled");
    return 0;

err_dma:
    if (npu->dma_chan)
        dma_release_channel(npu->dma_chan);
err_clk:
    if (npu->aclk)
        clk_disable_unprepare(npu->aclk);
err_decode_clk:
    if (npu->decode_clk)
        clk_disable_unprepare(npu->decode_clk);
err_prefill_clk:
    if (npu->prefill_clk)
        clk_disable_unprepare(npu->prefill_clk);
err_npu_clk:
    if (npu->npu_clk)
        clk_disable_unprepare(npu->npu_clk);
err_active:
    atomic_set(&active_devices, 0);
    return ret;
}

static void npu_remove(struct platform_device *pdev)
{
    struct npu_dev *npu = platform_get_drvdata(pdev);

    npu_unregister_chrdev(npu);
    if (npu->dma_chan)
        dma_release_channel(npu->dma_chan);

    if (npu->rst)
        reset_control_assert(npu->rst);
    if (npu->aclk)
        clk_disable_unprepare(npu->aclk);
    if (npu->decode_clk)
        clk_disable_unprepare(npu->decode_clk);
    if (npu->prefill_clk)
        clk_disable_unprepare(npu->prefill_clk);
    if (npu->npu_clk)
        clk_disable_unprepare(npu->npu_clk);
    atomic_set(&active_devices, 0);
}

static const struct of_device_id npu_of_match[] = {
    { .compatible = "xlnx,kv260-npu-generic-1.0" },
    { .compatible = "xlnx,T-NPU-FPGA-1.0" },
    { }
};
MODULE_DEVICE_TABLE(of, npu_of_match);

static struct platform_driver npu_driver = {
    .driver = {
        .name = DRIVER_NAME,
        .of_match_table = npu_of_match,
    },
    .probe = npu_probe,
    .remove_new = npu_remove,
};

static int __init npu_module_init(void)
{
    struct resource resources[2];
    int nresources = 0;
    int ret;

    ret = platform_driver_register(&npu_driver);
    if (ret)
        return ret;

    if (!reg_base)
        return 0;

    if (!reg_size) {
        ret = -EINVAL;
        goto err_driver;
    }

    memset(resources, 0, sizeof(resources));
    resources[nresources].start = reg_base;
    resources[nresources].end = reg_base + reg_size - 1;
    resources[nresources].flags = IORESOURCE_MEM;
    nresources++;

    if (irq >= 0) {
        resources[nresources].start = irq;
        resources[nresources].end = irq;
        resources[nresources].flags = IORESOURCE_IRQ;
        nresources++;
    }

    fallback_pdev = platform_device_register_resndata(NULL, DRIVER_NAME,
                                                      PLATFORM_DEVID_NONE,
                                                      resources, nresources,
                                                      NULL, 0);
    if (IS_ERR(fallback_pdev)) {
        ret = PTR_ERR(fallback_pdev);
        fallback_pdev = NULL;
        goto err_driver;
    }

    return 0;

err_driver:
    platform_driver_unregister(&npu_driver);
    return ret;
}

static void __exit npu_module_exit(void)
{
    if (fallback_pdev)
        platform_device_unregister(fallback_pdev);
    platform_driver_unregister(&npu_driver);
    mutex_lock(&shared_buffer.lock);
    if (shared_buffer.cpu_addr) {
        dma_free_coherent(shared_buffer.dev, shared_buffer.size,
                          shared_buffer.cpu_addr, shared_buffer.dma_addr);
        put_device(shared_buffer.dev);
        shared_buffer.cpu_addr = NULL;
        shared_buffer.dev = NULL;
        shared_buffer.dma_addr = 0;
        shared_buffer.size = 0;
    }
    mutex_unlock(&shared_buffer.lock);
}

module_init(npu_module_init);
module_exit(npu_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("VersaEdge");
MODULE_DESCRIPTION("KV260 NPU driver using Linux CMA/DMA coherent buffers");
