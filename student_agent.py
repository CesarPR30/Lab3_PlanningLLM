import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

EXAMPLES_PATH_DEFAULT = "Examples.json"

COLORS = {"red","blue","orange","yellow","green","purple","pink","black","white","gray","grey","brown"}

def _domain_of(s: str) -> str:
    s = s.lower()
    if "set of blocks" in s:
        return "blocks"
    if "set of objects" in s:
        return "objects"
    return "blocks" if "block" in s else "objects"

def _extract_last_unsolved_statement(scenario_context: str) -> str:
    sc = scenario_context
    last_stmt = sc.rfind("[STATEMENT]")
    if last_stmt == -1:
        return sc
    tail = sc[last_stmt:]
    last_plan = tail.rfind("[PLAN]")
    if last_plan == -1:
        return tail
    return tail[: last_plan + len("[PLAN]")]

_token_re = re.compile(r"[a-zA-Z_]+|\d+")

def _tokenize(text: str) -> List[str]:
    toks = _token_re.findall(text.lower())
    out: List[str] = []
    for t in toks:
        out.append(t)
        if t in COLORS:
            out.extend([t, t])
        if len(t) == 1 and t.isalpha(): 
            out.extend([t, t])
    return out

def _bag_overlap_score(q: List[str], d: List[str]) -> float:
    from collections import Counter
    cq, cd = Counter(q), Counter(d)
    inter = 0
    union = 0
    for k, v in cq.items():
        inter += min(v, cd.get(k, 0))
        union += max(v, cd.get(k, 0))
    for k, v in cd.items():
        if k not in cq:
            union += v
    return (inter / union) if union else 0.0

_goal_blocks_re = re.compile(r"my goal is to have that (.+?)\.", re.IGNORECASE | re.DOTALL)
_on_top_re = re.compile(r"the ([a-z]+) block is on top of the ([a-z]+) block", re.IGNORECASE)

def _extract_block_goals(problem: str) -> List[Tuple[str,str]]:
    m = _goal_blocks_re.search(problem)
    if not m:
        return []
    goal_txt = m.group(1)
    pairs = _on_top_re.findall(goal_txt)
    return [(a.lower(), b.lower()) for a,b in pairs]

_goal_obj_re = re.compile(r"my goal is to have that (.+?)\.", re.IGNORECASE | re.DOTALL)
_craves_re = re.compile(r"object ([a-z]) craves object ([a-z])", re.IGNORECASE)

def _extract_object_goals(problem: str) -> List[Tuple[str,str]]:
    m = _goal_obj_re.search(problem)
    if not m:
        return []
    goal_txt = m.group(1)
    pairs = _craves_re.findall(goal_txt)
    return [(a.lower(), b.lower()) for a,b in pairs]

_ACTION_RE = re.compile(r"^\([a-zA-Z_]+\s+[a-zA-Z0-9_]+\s*(?:[a-zA-Z0-9_]+)?\)$")

def validate_plan(domain: str, plan: List[str]) -> List[str]:
    issues: List[str] = []
    if not plan:
        return ["EMPTY_PLAN"]

    allowed = {"unmount_node","mount_node","engage_payload","release_payload"} if domain=="blocks" else {"attack","succumb","feast","overcome"}

    for i, line in enumerate(plan):
        if "from" in line.lower():
            issues.append(f"LINE_{i}_CONTAINS_FROM")
        if not (line.startswith("(") and line.endswith(")")):
            issues.append(f"LINE_{i}_NOT_PARENTHESIZED")
            continue
        if not _ACTION_RE.match(line):
            issues.append(f"LINE_{i}_BAD_FORMAT")
        inner = line[1:-1].strip().split()
        if not inner:
            issues.append(f"LINE_{i}_EMPTY")
            continue
        act = inner[0]
        if act not in allowed:
            issues.append(f"LINE_{i}_UNKNOWN_ACTION:{act}")
        if domain=="blocks":
            if act in {"unmount_node","mount_node"} and len(inner)!=3:
                issues.append(f"LINE_{i}_BAD_ARITY:{act}")
            if act in {"engage_payload","release_payload"} and len(inner)!=2:
                issues.append(f"LINE_{i}_BAD_ARITY:{act}")
        else:
            if act in {"attack","succumb"} and len(inner)!=2:
                issues.append(f"LINE_{i}_BAD_ARITY:{act}")
            if act in {"feast","overcome"} and len(inner)!=3:
                issues.append(f"LINE_{i}_BAD_ARITY:{act}")
    return issues

class AssemblyAgent:
    def __init__(self, examples_path: str = EXAMPLES_PATH_DEFAULT, shots_k: int = 6):
        self.examples_path = examples_path
        self.shots_k = shots_k

        self.system_objects = (
            "You are an expert planner in the OBJECTS domain. "
            "Output ONLY the final plan as parenthesized action lines. "
            "No explanations, no markdown, no extra text."
        )
        self.system_blocks = (
            "You are an expert planner in the BLOCKS domain. "
            "Output ONLY the final plan as parenthesized action lines. "
            "No explanations, no markdown, no extra text."
        )

        self.blocks: List[Dict] = []
        self.objects: List[Dict] = []

        p = Path(self.examples_path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for ex in data:
                dom = _domain_of(ex["scenario_context"])
                prob = _extract_last_unsolved_statement(ex["scenario_context"])
                item = {
                    "problem": prob,
                    "target_action_sequence": ex["target_action_sequence"],
                    "tok": _tokenize(prob),
                    "L": len(ex["target_action_sequence"]),
                    "goals_blocks": _extract_block_goals(prob),
                    "goals_objects": _extract_object_goals(prob),
                }
                (self.blocks if dom=="blocks" else self.objects).append(item)

    def _select_shots(self, problem: str, domain: str) -> List[Dict]:
        pool = self.blocks if domain=="blocks" else self.objects
        if not pool:
            return []

        qtok = _tokenize(problem)
        if domain=="blocks":
            qgoals = _extract_block_goals(problem)
        else:
            qgoals = _extract_object_goals(problem)

        scored = []
        for ex in pool:
            base = _bag_overlap_score(qtok, ex["tok"])
            boost = 0.0
            if domain=="blocks":
                eg = ex["goals_blocks"]
                if qgoals and eg and len(qgoals)==len(eg):
                    qx = {a for a,_ in qgoals}
                    exx = {a for a,_ in eg}
                    if qx & exx:
                        boost += 0.08
            else:
                eg = ex["goals_objects"]
                if qgoals and eg and len(qgoals)==len(eg):
                    qx = {a for a,_ in qgoals}
                    exx = {a for a,_ in eg}
                    if qx & exx:
                        boost += 0.08
            scored.append((base + boost, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[: self.shots_k]]

    def _build_prompt_objects(self, problem: str, shots: List[Dict]) -> str:
        goals = _extract_object_goals(problem)
        goal_hint = ""
        if goals:
            goal_hint = "GOAL CRAVINGS (must satisfy exactly): " + ", ".join([f"{a}->{b}" for a,b in goals]) + "\n"

        parts = []
        parts.append(
            "OUTPUT FORMAT (STRICT):\n"
            "- Output ONLY action lines.\n"
            "- One action per line.\n"
            "- Each line MUST start with '(' and end with ')'.\n"
            "- NEVER write the word 'from' inside parentheses.\n\n"
            "VALID ACTIONS:\n"
            "- (attack X)\n"
            "- (succumb X)\n"
            "- (feast X Y)\n"
            "- (overcome X Y)\n\n"
            "STRICT PLANNING RULES:\n"
            "1) Use ONLY the LAST [STATEMENT] initial conditions + goal.\n"
            "2) Only act on object letters that appear in the GOAL cravings.\n"
            "3) If GOAL is a single craving X craves Y, prefer EXACTLY 2 steps:\n"
            "   (attack X)\n"
            "   (overcome X Y)\n"
            "   unless Attack is impossible due to missing preconditions.\n"
            "4) Do NOT use feast/succumb if Attack is already possible.\n"
            "5) Never add extra actions after goal is satisfied.\n"
            "6) Overcome order: if you output (overcome X Y), you must have (attack X) earlier\n"
            "   unless Pain X is explicitly already true.\n\n"
            + goal_hint +
            "Now solve. Output plan lines only.\n"
        )

        if shots:
            parts.append("\nEXAMPLES (copy the style and minimality):")
            for ex in shots[:3]:
                parts.append("\nProblem:\n" + ex["problem"])
                parts.append("Plan:")
                parts.extend(ex["target_action_sequence"])
            parts.append("\nEND EXAMPLES.\n")

        parts.append("PROBLEM:\n" + problem)
        parts.append("\nPLAN:")
        return "\n".join(parts)

    def _build_prompt_blocks(self, problem: str, shots: List[Dict]) -> str:
        goals = _extract_block_goals(problem)
        goal_hint = ""
        if goals:
            goal_hint = "GOAL STACKS (must satisfy): " + ", ".join([f"{a} on {b}" for a,b in goals]) + "\n"

        parts = []
        parts.append(
            "OUTPUT FORMAT (STRICT):\n"
            "- Output ONLY action lines.\n"
            "- One action per line.\n"
            "- Each line MUST start with '(' and end with ')'.\n\n"
            "VALID ACTIONS:\n"
            "- (unmount_node X Y)\n"
            "- (mount_node X Y)\n"
            "- (engage_payload X)\n"
            "- (release_payload X)\n\n"
            "MECHANICS (STRICT):\n"
            "A) unmount_node X Y means X was on Y; after this you are HOLDING X.\n"
            "B) mount_node X Y requires you are holding X; after this hand becomes EMPTY.\n"
            "C) engage_payload X only for blocks ON THE TABLE; after this you are HOLDING X.\n"
            "D) release_payload X puts down held X; after this hand becomes EMPTY.\n\n"
            "STRICT OPTIMALITY RULES:\n"
            "1) If goal includes 'X on Y', the final stacking for that relation MUST be (mount_node X Y).\n"
            "2) If X is currently on Z and you need X on Y, the core move is:\n"
            "   (unmount_node X Z)\n"
            "   (mount_node X Y)\n"
            "   Do NOT insert release/engage between them.\n"
            "3) Use release/engage ONLY when you must switch which block you are holding.\n"
            "4) Never use engage_payload before an unmount_node.\n"
            "5) Build towers bottom-up.\n"
            "6) Use the SHORTEST valid plan.\n\n"
            + goal_hint +
            "Now solve. Output plan lines only.\n"
        )

        if shots:
            parts.append("\nEXAMPLES (copy the minimal patterns):")
            for ex in shots[:3]:
                parts.append("\nProblem:\n" + ex["problem"])
                parts.append("Plan:")
                parts.extend(ex["target_action_sequence"])
            parts.append("\nEND EXAMPLES.\n")

        parts.append("PROBLEM:\n" + problem)
        parts.append("\nPLAN:")
        return "\n".join(parts)

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        domain = _domain_of(scenario_context)
        problem = _extract_last_unsolved_statement(scenario_context)

        shots = self._select_shots(problem, domain)

        if domain == "blocks":
            system = self.system_blocks
            prompt = self._build_prompt_blocks(problem, shots)
            max_tokens = 192
        else:
            system = self.system_objects
            prompt = self._build_prompt_objects(problem, shots)
            max_tokens = 192

        resp = llm_engine_func(
            prompt=prompt,
            system=system,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=max_tokens,
            enable_thinking=False,
            stream=False
        )

        plan = [l.strip() for l in resp.split("\n") if l.strip()]

        debug_validate = False
        if debug_validate:
            issues = validate_plan(domain, plan)
            if issues:
                print("[VALIDATION WARNINGS]", issues[:12])

        return plan