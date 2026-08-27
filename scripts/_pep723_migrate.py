#!/usr/bin/env python3
from __future__ import annotations
import ast, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SELF=Path(__file__).resolve()
SKIP={".git",".github",".venv","venv","src","tests","test","scripts","__pycache__"}
MAP={"numpy":"numpy"}
def has_main(tree):
    for node in tree.body:
        if isinstance(node,ast.If) and isinstance(node.test,ast.Compare):
            if "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test): return True
    return False
def main():
    unknown=[]; plans=[]
    for path in sorted(ROOT.rglob("*.py")):
        rel=path.relative_to(ROOT)
        if path==SELF or any(part in SKIP for part in rel.parts) or path.name.startswith("test_"): continue
        text=path.read_text(encoding="utf-8"); tree=ast.parse(text)
        first=text.splitlines()[0] if text.splitlines() else ""
        if not (has_main(tree) or (first.startswith("#!") and "python" in first)): continue
        imports=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module: imports.add(node.module.split('.')[0])
        ext=imports-set(sys.stdlib_module_names); deps=[]
        for module in sorted(ext):
            if module not in MAP: unknown.append(f"{rel}: unknown import {module}")
            else: deps.append(MAP[module])
        block=['# /// script','# requires-python = ">=3.11"','# dependencies = [',*[f'#     "{d}",' for d in deps],'# ]','# ///']
        lines=text.splitlines(keepends=True)
        if lines and lines[0].startswith('#!'): lines[0]='#!/usr/bin/env -S uv run --script\n'
        else: lines.insert(0,'#!/usr/bin/env -S uv run --script\n')
        rebuilt=''.join(lines); firstline,_,rest=rebuilt.partition('\n')
        plans.append((path,f"{firstline}\n#\n"+'\n'.join(block)+f"\n{rest}"))
    if unknown:
        print('\n'.join(unknown),file=sys.stderr); return 2
    for path,text in plans:
        ast.parse(text); path.write_text(text,encoding='utf-8',newline='\n'); print(path.relative_to(ROOT))
    return 0
if __name__=='__main__': raise SystemExit(main())
