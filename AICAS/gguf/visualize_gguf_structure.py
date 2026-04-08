import argparse
import html
import math
import struct
from pathlib import Path


GGUF_TYPES = {
    0: ("UINT8", "B"),
    1: ("INT8", "b"),
    2: ("UINT16", "H"),
    3: ("INT16", "h"),
    4: ("UINT32", "I"),
    5: ("INT32", "i"),
    6: ("FLOAT32", "f"),
    7: ("BOOL", "?"),
    8: ("STRING", None),
    9: ("ARRAY", None),
    10: ("UINT64", "Q"),
    11: ("INT64", "q"),
    12: ("FLOAT64", "d"),
}


def read_u32(handle):
    return struct.unpack("<I", handle.read(4))[0]


def read_u64(handle):
    return struct.unpack("<Q", handle.read(8))[0]


def read_string(handle):
    length = read_u64(handle)
    return handle.read(length).decode("utf-8", errors="replace")


def read_scalar(handle, value_type):
    _, fmt = GGUF_TYPES[value_type]
    if value_type == 8:
        return read_string(handle)
    if value_type == 9:
        element_type = read_u32(handle)
        length = read_u64(handle)
        return [read_scalar(handle, element_type) for _ in range(length)]
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, handle.read(size))[0]


def parse_gguf(path):
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32(handle)
        tensor_count = read_u64(handle)
        metadata_count = read_u64(handle)
        metadata = {}
        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = read_u32(handle)
            metadata[key] = read_scalar(handle, value_type)

        tensors = []
        for _ in range(tensor_count):
            name = read_string(handle)
            rank = read_u32(handle)
            dims = [read_u64(handle) for _ in range(rank)]
            ggml_type = read_u32(handle)
            _offset = read_u64(handle)
            tensors.append({"name": name, "dims": dims, "type": ggml_type})

    return {"version": version, "metadata": metadata, "tensors": tensors}


def param_count(tensors):
    return sum(math.prod(tensor["dims"]) for tensor in tensors)


def format_count(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def format_dims(dims):
    return "x".join(str(v) for v in dims)


def escape(text):
    return html.escape(str(text))


def llama_section(name, parsed):
    meta = parsed["metadata"]
    tensors = parsed["tensors"]
    block_count = int(meta.get("llama.block_count", 0))
    head_count = int(meta.get("llama.attention.head_count", 0))
    kv_head_count = int(meta.get("llama.attention.head_count_kv", 0))
    embd = int(meta.get("llama.embedding_length", 0))
    ffn = int(meta.get("llama.feed_forward_length", 0))
    ctx = int(meta.get("llama.context_length", 0))
    vocab = int(meta.get("llama.vocab_size", 0))

    return f"""
    <section class="panel">
      <h2>{escape(name)}<span class="pill">language backbone</span></h2>
      <p class="meta">
        arch=<strong>{escape(meta.get("general.architecture", "?"))}</strong>
        · blocks=<strong>{block_count}</strong>
        · hidden=<strong>{embd}</strong>
        · ffn=<strong>{ffn}</strong>
        · heads=<strong>{head_count}</strong>
        · kv-heads=<strong>{kv_head_count}</strong>
        · ctx=<strong>{ctx}</strong>
        · vocab=<strong>{vocab}</strong>
        · params/elements=<strong>{format_count(param_count(tensors))}</strong>
      </p>
      <div class="flow">
        <div class="node wide">
          <div class="title">token_embd</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "token_embd.weight"))}] Q8_0</div>
        </div>
        <div class="arrow">→</div>
        <div class="stack">
          <div class="stack-count">{block_count}× decoder block</div>
          <div class="block-grid">
            <div class="node">attn_norm</div>
            <div class="node">attn_q</div>
            <div class="node">attn_k</div>
            <div class="node">attn_v</div>
            <div class="node">attn_output</div>
            <div class="node">ffn_norm</div>
            <div class="node">ffn_gate</div>
            <div class="node">ffn_up</div>
            <div class="node">ffn_down</div>
          </div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <div class="title">output_norm</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "output_norm.weight"))}] F32</div>
        </div>
        <div class="arrow">→</div>
        <div class="node wide">
          <div class="title">lm_head / output</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "output.weight"))}] Q8_0</div>
        </div>
      </div>
    </section>
    """


def clip_section(name, parsed):
    meta = parsed["metadata"]
    tensors = parsed["tensors"]
    block_count = int(meta.get("clip.vision.block_count", 0))
    embd = int(meta.get("clip.vision.embedding_length", 0))
    ffn = int(meta.get("clip.vision.feed_forward_length", 0))
    heads = int(meta.get("clip.vision.attention.head_count", 0))
    image_size = int(meta.get("clip.vision.image_size", 0))
    patch = int(meta.get("clip.vision.patch_size", 0))
    projection = int(meta.get("clip.vision.projection_dim", 0))
    projector_scale = meta.get("clip.vision.projector.scale_factor", "?")

    return f"""
    <section class="panel">
      <h2>{escape(name)}<span class="pill">vision tower + projector</span></h2>
      <p class="meta">
        arch=<strong>{escape(meta.get("general.architecture", "?"))}</strong>
        · vision-blocks=<strong>{block_count}</strong>
        · hidden=<strong>{embd}</strong>
        · ffn=<strong>{ffn}</strong>
        · heads=<strong>{heads}</strong>
        · image=<strong>{image_size}</strong>
        · patch=<strong>{patch}</strong>
        · projection=<strong>{projection}</strong>
        · projector-scale=<strong>{escape(projector_scale)}</strong>
        · params/elements=<strong>{format_count(param_count(tensors))}</strong>
      </p>
      <div class="flow">
        <div class="node">
          <div class="title">image input</div>
          <div class="sub">{image_size}x{image_size} RGB</div>
        </div>
        <div class="arrow">→</div>
        <div class="node wide">
          <div class="title">patch_embd</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "v.patch_embd.weight"))}] F32</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <div class="title">position_embd</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "v.position_embd.weight"))}] F32</div>
        </div>
        <div class="arrow">→</div>
        <div class="stack">
          <div class="stack-count">{block_count}× vision block</div>
          <div class="block-grid">
            <div class="node">ln1</div>
            <div class="node">attn_q</div>
            <div class="node">attn_k</div>
            <div class="node">attn_v</div>
            <div class="node">attn_out</div>
            <div class="node">ln2</div>
            <div class="node">ffn_down</div>
            <div class="node">ffn_up</div>
          </div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <div class="title">post_ln</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "v.post_ln.weight"))}] F32</div>
        </div>
        <div class="arrow">→</div>
        <div class="node wide">
          <div class="title">mm.model.fc</div>
          <div class="sub">[{format_dims(find_tensor_dims(tensors, "mm.model.fc.weight"))}] Q8_0</div>
        </div>
      </div>
    </section>
    """


def find_tensor_dims(tensors, tensor_name):
    for tensor in tensors:
        if tensor["name"] == tensor_name:
            return tensor["dims"]
    return ["?"]


def build_html(main_name, main_parsed, mmproj_name=None, mmproj_parsed=None):
    title = escape(main_parsed["metadata"].get("general.name", main_name))
    sections = [llama_section(main_name, main_parsed)]
    bridge = ""
    if mmproj_parsed is not None:
        sections.insert(0, clip_section(mmproj_name, mmproj_parsed))
        bridge = """
        <section class="bridge">
          <div class="bridge-box">
            Visual tokens from <strong>mmproj</strong> are projected to width <strong>960</strong>
            and then injected into the language backbone as multimodal input.
          </div>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} structure</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf2;
      --ink: #1f1f1f;
      --muted: #665f57;
      --line: #d8ccbd;
      --accent: #0e6b5c;
      --accent-2: #ba5d2a;
      --stack: #efe4d7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(186, 93, 42, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 107, 92, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f1e9 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
    }}
    .lead {{
      margin: 0 0 24px;
      color: var(--muted);
      max-width: 860px;
      font-size: 18px;
      line-height: 1.45;
    }}
    .panel {{
      margin: 24px 0;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.05);
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 28px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid var(--line);
      color: var(--accent);
      background: #f3fbf8;
    }}
    .meta {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .flow {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    .node, .stack {{
      min-width: 140px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: white;
    }}
    .node {{
      text-align: center;
    }}
    .wide {{
      min-width: 210px;
    }}
    .title {{
      font-weight: 700;
      font-size: 16px;
    }}
    .sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .arrow {{
      font-size: 28px;
      color: var(--accent-2);
      line-height: 1;
    }}
    .stack {{
      background: var(--stack);
      min-width: min(100%, 420px);
      flex: 1 1 360px;
    }}
    .stack-count {{
      margin-bottom: 12px;
      font-weight: 700;
      color: var(--accent);
    }}
    .block-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
    }}
    .bridge {{
      margin: 14px 0 6px;
    }}
    .bridge-box {{
      padding: 18px 20px;
      border-left: 6px solid var(--accent-2);
      background: rgba(186, 93, 42, 0.08);
      border-radius: 12px;
      line-height: 1.5;
    }}
    .footnote {{
      margin-top: 26px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    code {{
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 0.95em;
    }}
    @media (max-width: 720px) {{
      .flow {{
        flex-direction: column;
        align-items: stretch;
      }}
      .arrow {{
        align-self: center;
        transform: rotate(90deg);
      }}
      .stack {{
        min-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p class="lead">
      This page is a static structure view generated from GGUF metadata and tensor names.
      It shows the model hierarchy stored in the files, not the full per-token runtime ggml compute graph.
    </p>
    {''.join(sections)}
    {bridge}
    <p class="footnote">
      If you need the actual runtime graph used by <code>llama.cpp</code>/<code>ggml</code>,
      the usual path is <code>ggml_graph_print()</code> or <code>ggml_graph_dump_dot()</code>
      inside a debug build, then render the exported DOT with Graphviz.
    </p>
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Visualize GGUF structure as HTML.")
    parser.add_argument("--main", required=True, help="Main model GGUF path")
    parser.add_argument("--mmproj", help="Optional mmproj GGUF path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML path",
    )
    args = parser.parse_args()

    main_path = Path(args.main)
    output_path = Path(args.output)
    mmproj_path = Path(args.mmproj) if args.mmproj else None

    main_parsed = parse_gguf(main_path)
    mmproj_parsed = parse_gguf(mmproj_path) if mmproj_path else None

    html_text = build_html(
        main_path.name,
        main_parsed,
        mmproj_path.name if mmproj_path else None,
        mmproj_parsed,
    )
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
