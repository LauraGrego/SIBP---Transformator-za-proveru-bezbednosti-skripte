from __future__ import annotations

import re


MALICIOUS_RULES = (
    (
        "reverse_shell",
        re.compile(r"(?:bash|sh)\s+-i[^\n]*(?:/dev/(?:tcp|udp)|>&)", re.IGNORECASE),
        "interactive shell connected through a network device",
    ),
    (
        "reverse_shell",
        re.compile(r"\b(?:nc|netcat)\b[^\n]*\s-e\s+(?:/bin/)?(?:ba)?sh\b", re.IGNORECASE),
        "netcat launches a command shell for a remote connection",
    ),
    (
        "exfiltration",
        re.compile(
            r"\bcurl\b[^\n]*(?:--data-binary|--upload-file|-t|-f)\s+@?(?:/etc/(?:passwd|shadow)|/home/|/root/)",
            re.IGNORECASE,
        ),
        "uploads a sensitive local file to a remote endpoint",
    ),
    (
        "persistence",
        re.compile(
            r"(?:curl|wget)[^\n|]*(?:\||\n)[^\n]*(?:ba)?sh[^\n]*(?:\||\n)[^\n]*crontab|crontab[^\n]*(?:curl|wget)",
            re.IGNORECASE,
        ),
        "installs a remotely downloaded command in cron",
    ),
)

RISKY_RULES = (
    (
        "firewall_change",
        re.compile(r"\biptables\s+-F\b|\bufw\s+(?:allow|disable)\b", re.IGNORECASE),
        "changes or flushes host firewall rules",
    ),
    (
        "firewall_change",
        re.compile(r"\bfirewall-cmd\b[^\n]*--(?:permanent|add-port|add-rich-rule)", re.IGNORECASE),
        "makes a persistent firewall configuration change",
    ),
    (
        "destructive_file_operation",
        re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f\b\s+/(?:etc|var|usr|home|root)\b", re.IGNORECASE),
        "recursively deletes files from a system directory",
    ),
)

RISK_EXPLANATION_RULES = (
    (
        "account_security_change",
        re.compile(r"\b(?:passwd|useradd|userdel|usermod|groupadd|groupdel)\b", re.IGNORECASE),
        "changes user, group, password, or account security configuration",
    ),
    (
        "permission_change",
        re.compile(r"\b(?:chmod|chown|chgrp|setfacl)\b", re.IGNORECASE),
        "changes file ownership or access permissions",
    ),
    (
        "service_management",
        re.compile(r"\b(?:systemctl|service)\s+(?:start|stop|restart|enable|disable)\b", re.IGNORECASE),
        "changes the state or startup behavior of a system service",
    ),
    (
        "package_management",
        re.compile(r"\b(?:apt-get|apt|dnf|yum|pacman|zypper)\s+(?:install|remove|upgrade|update)\b", re.IGNORECASE),
        "installs, removes, or updates system packages",
    ),
    (
        "privileged_operation",
        re.compile(r"\b(?:sudo|su)\b", re.IGNORECASE),
        "executes commands with elevated privileges",
    ),
)

INFINITE_LOOP_PATTERNS = (
    re.compile(
        r"\bwhile\s+(?:true|:)\s*;?\s*do(?P<body>.*?)\bdone\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bfor\s*\(\(\s*;\s*;\s*\)\)\s*;?\s*do(?P<body>.*?)\bdone\b",
        re.IGNORECASE | re.DOTALL,
    ),
)

LOOP_ESCAPE_PATTERN = re.compile(
    r"\b(?:break|exit|return|sleep|read|wait)\b", re.IGNORECASE
)

DNS_DIAGNOSTIC_BLOCKLIST = re.compile(
    r"\b(?:sudo|curl|wget|nc|netcat|nmap|masscan|rm|chmod|chown|dd|mkfs|mount|"
    r"iptables|ufw|firewall-cmd|systemctl|service|crontab|at|ssh|scp|eval|exec)\b",
    re.IGNORECASE,
)


def validate_script_text(script: str) -> str | None:
    """Reject empty or tabular inputs that are not one shell script."""
    if not script.strip():
        return "empty file"
    firstLine = script.lstrip().splitlines()[0].lower()
    if firstLine.startswith("script_id,") or firstLine.startswith("id,script"):
        return "tabular dataset, not a single shell script"
    return None


def find_security_rule(script: str) -> dict | None:
    """Return a high-confidence result for explicit dangerous shell behavior."""
    for reason, pattern, explanation in MALICIOUS_RULES:
        if pattern.search(script):
            return {
                "label": "malicious",
                "confidence": 0.999,
                "reason": reason,
                "reason_confidence": 0.999,
                "explanation": explanation,
                "source": "security_rule",
            }
    if contains_unbounded_busy_loop(script):
        return {
            "label": "risky",
            "confidence": 0.99,
            "reason": "infinite_loop",
            "explanation": (
                "contains an unbounded loop without an obvious pause or exit path"
            ),
            "source": "security_rule",
        }
    for reason, pattern, explanation in RISKY_RULES:
        if pattern.search(script):
            return {
                "label": "risky",
                "confidence": 0.99,
                "reason": reason,
                "explanation": explanation,
                "source": "security_rule",
            }
    if is_read_only_dns_diagnostic(script):
        return {
            "label": "safe",
            "confidence": 0.99,
            "reason": "read_only_dns_diagnostic",
            "explanation": "performs read-only DNS queries without mutating commands",
            "source": "security_rule",
        }
    return None


def contains_unbounded_busy_loop(script: str) -> bool:
    """Detect common infinite busy loops while excluding obvious safe controls."""
    for pattern in INFINITE_LOOP_PATTERNS:
        for match in pattern.finditer(script):
            loopBody = match.group("body")
            # Words printed to users (for example, "no exit condition") are
            # not control-flow statements, so remove strings and comments
            # before looking for an escape or blocking command.
            controlBody = re.sub(r"(['\"]).*?\1", "", loopBody, flags=re.DOTALL)
            controlBody = re.sub(r"(?m)#.*$", "", controlBody)
            if not LOOP_ESCAPE_PATTERN.search(controlBody):
                return True
    return False


def is_read_only_dns_diagnostic(script: str) -> bool:
    """Recognize DNS query utilities only when no dangerous command is present."""
    digCalls = re.findall(r"\bdig\s+", script, re.IGNORECASE)
    if len(digCalls) < 2:
        return False
    commandText = re.sub(r"(['\"]).*?\1", "", script, flags=re.DOTALL)
    commandText = re.sub(r"(?m)#.*$", "", commandText)
    return DNS_DIAGNOSTIC_BLOCKLIST.search(commandText) is None


def explain_risky_behavior(script: str) -> dict:
    """Return the clearest available reason for a model-produced risky label."""
    for reason, pattern, explanation in (*RISKY_RULES, *RISK_EXPLANATION_RULES):
        if pattern.search(script):
            return {"reason": reason, "explanation": explanation}
    return {
        "reason": "model_detected_risk",
        "explanation": (
            "the statistical model found risky patterns, but no specific "
            "high-confidence operation was identified"
        ),
    }
