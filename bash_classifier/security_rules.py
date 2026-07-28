from __future__ import annotations

import re
import shlex


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
            r"(?i:\bcurl\b)[^\n]*"
            r"(?:(?i:--data(?:-binary)?|--upload-file)(?:=|\s)+@?"
            r"|-[dT]\s*@?"
            r"|(?i:--form)(?:=|\s)+[^\s=]+=@?"
            r"|-F\s+[^\s=]+=@?)"
            r"(?:/etc/(?:passwd|shadow|sudoers)|/home/|/root/)",
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

NON_EXECUTING_OUTPUT_COMMANDS = {"echo", "printf"}
CONTROL_PREFIXES = {"if", "then", "elif", "else", "while", "until", "do", "!"}
COMMAND_WRAPPERS = {"command", "env", "nohup", "sudo"}
DNS_ALLOWED_COMMANDS = {
    "[",
    "[[",
    ":",
    "date",
    "dig",
    "echo",
    "head",
    "local",
    "printf",
    "return",
    "true",
}
DNS_DISALLOWED_QUERY_OPTIONS = {"axfr", "ixfr", "-f"}
EXECUTED_OUTPUT_PATTERN = re.compile(
    r"\becho\b[^\n]*?(['\"])(?P<body>.*?)\1\s*\)?\s*\|\s*"
    r"(?P<sink>bash|sh|crontab|at|eval)\b",
    re.IGNORECASE,
)
HEREDOC_START_PATTERN = re.compile(
    r"^\s*(?P<command>[A-Za-z_][A-Za-z0-9_.-]*)\b.*?"
    r"<<(?P<strip_tabs>-?)\s*['\"]?(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)",
)
HEREDOC_EXECUTION_COMMANDS = {"bash", "sh", "eval"}


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
    executableText = executable_rule_text(script)
    for reason, pattern, explanation in MALICIOUS_RULES:
        if pattern.search(executableText):
            return {
                "label": "malicious",
                "confidence": 0.999,
                "reason": reason,
                "reason_confidence": 0.999,
                "explanation": explanation,
                "source": "security_rule",
            }
    if contains_unbounded_busy_loop(remove_inert_heredocs(script)):
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
        if pattern.search(executableText):
            return {
                "label": "risky",
                "confidence": 0.99,
                "reason": reason,
                "explanation": explanation,
                "source": "security_rule",
            }
    if is_strict_read_only_dns_diagnostic(script):
        return {
            "label": "safe",
            "confidence": 0.99,
            "reason": "read_only_dns_diagnostic",
            "explanation": (
                "contains only allowlisted shell helpers and read-only DNS queries"
            ),
            "source": "security_rule",
        }
    return None


def executable_rule_text(script: str) -> str:
    """Return command text while excluding comments and inert output strings."""
    script = remove_inert_heredocs(script)
    try:
        segments = shell_segments(script)
    except ValueError:
        # An incomplete script should still be inspected, but comments must not
        # become executable signatures.
        return re.sub(r"(?m)#.*$", "", script)

    executableSegments = [
        f"{match.group('body')}\n{match.group('sink')}"
        for match in EXECUTED_OUTPUT_PATTERN.finditer(script)
        if not _line_is_comment(script, match.start())
    ]
    for current in segments:
        commandIndex = _command_index(current)
        if commandIndex is None:
            continue
        command = current[commandIndex].lower()
        if command in NON_EXECUTING_OUTPUT_COMMANDS:
            substitutions = [
                token
                for token in current[commandIndex + 1 :]
                if "$(" in token or "`" in token
            ]
            executableSegments.extend(substitutions)
            continue
        executableSegments.append(" ".join(current))
    return "\n".join(executableSegments)


def shell_segments(script: str) -> list[list[str]]:
    """Tokenize shell text into command-like segments without executing it."""
    lexer = shlex.shlex(
        script,
        posix=True,
        punctuation_chars=";&|()\n<>",
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = "#"
    segments: list[list[str]] = []
    segment: list[str] = []
    for token in lexer:
        if _is_command_separator(token):
            if segment:
                segments.append(segment)
                segment = []
        else:
            segment.append(token)
    if segment:
        segments.append(segment)
    return segments


def _is_command_separator(token: str) -> bool:
    """Recognize operator runs emitted by shlex while retaining redirections."""
    if token in {";", ";;", "&", "&&", "|", "||", "(", ")"}:
        return True
    return "\n" in token and not token.strip(";&|()\n")


def _line_is_comment(script: str, position: int) -> bool:
    """Return whether a regex match starts after a comment marker on its line."""
    lineStart = script.rfind("\n", 0, position) + 1
    prefix = script[lineStart:position]
    quote = None
    escaped = False
    for character in prefix:
        if escaped:
            escaped = False
        elif character == "\\" and quote != "'":
            escaped = True
        elif quote is None and character in {"'", '"'}:
            quote = character
        elif character == quote:
            quote = None
        elif character == "#" and quote is None:
            return True
    return False


def remove_inert_heredocs(script: str) -> str:
    """Blank heredoc data unless it is explicitly supplied to a shell evaluator."""
    lines = script.splitlines(keepends=True)
    result: list[str] = []
    delimiter = None
    stripTabs = False
    for line in lines:
        if delimiter is not None:
            candidate = line.rstrip("\r\n")
            if stripTabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                delimiter = None
                stripTabs = False
                result.append("\n" if line.endswith("\n") else "")
            else:
                result.append("\n" if line.endswith("\n") else "")
            continue

        match = HEREDOC_START_PATTERN.search(line)
        if (
            match is not None
            and match.group("command").lower() not in HEREDOC_EXECUTION_COMMANDS
        ):
            delimiter = match.group("delimiter")
            stripTabs = bool(match.group("strip_tabs"))
        result.append(line)
    return "".join(result)


def _command_index(tokens: list[str]) -> int | None:
    """Find the effective command token after shell control words and wrappers."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        if lowered in CONTROL_PREFIXES or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", token
        ):
            index += 1
            continue
        if lowered in COMMAND_WRAPPERS:
            index += 1
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue
        if lowered in {"fi", "done", "esac", "{", "}"}:
            return None
        return index
    return None


def is_strict_read_only_dns_diagnostic(script: str) -> bool:
    """Allowlist complete DNS diagnostics without trusting an open-ended blocklist."""
    script = remove_inert_heredocs(script)
    if EXECUTED_OUTPUT_PATTERN.search(script):
        return False
    try:
        segments = shell_segments(script)
    except ValueError:
        return False

    digCalls = 0
    for tokens in segments:
        if _is_function_declaration(tokens):
            continue
        commandIndex = _command_index(tokens)
        if commandIndex is None:
            continue
        command = tokens[commandIndex].lower()

        if "sudo" in (token.lower() for token in tokens[: commandIndex + 1]):
            return False
        if command == "dig":
            if not _is_command_lookup(tokens, commandIndex):
                digCalls += 1
            loweredTokens = {token.lower() for token in tokens[commandIndex + 1 :]}
            if loweredTokens & DNS_DISALLOWED_QUERY_OPTIONS:
                return False
        elif command not in DNS_ALLOWED_COMMANDS:
            return False

        if not _dns_command_arguments_are_safe(command, tokens[commandIndex + 1 :]):
            return False
        if not _has_only_discarding_redirections(tokens):
            return False
        if command in NON_EXECUTING_OUTPUT_COMMANDS:
            for substitution in _command_substitutions(tokens[commandIndex + 1 :]):
                nestedSegments = shell_segments(substitution)
                for nested in nestedSegments:
                    nestedIndex = _command_index(nested)
                    if (
                        nestedIndex is not None
                        and nested[nestedIndex].lower() not in DNS_ALLOWED_COMMANDS
                    ):
                        return False
    return digCalls >= 2


def _is_command_lookup(tokens: list[str], commandIndex: int) -> bool:
    """Distinguish `command -v dig` discovery from an actual DNS query."""
    loweredPrefix = [token.lower() for token in tokens[:commandIndex]]
    return "command" in loweredPrefix and any(
        token in {"-v", "-V"} for token in tokens[:commandIndex]
    )


def _dns_command_arguments_are_safe(command: str, arguments: list[str]) -> bool:
    """Restrict allowlisted helpers that can also read files or change system time."""
    commandArguments = []
    for index, token in enumerate(arguments):
        if token in {">", ">>", ">&", "&>", "<", "<<"}:
            break
        if (
            token.isdigit()
            and index + 1 < len(arguments)
            and arguments[index + 1] in {">", ">>", ">&", "&>"}
        ):
            break
        commandArguments.append(token)
    if command == "head":
        return bool(
            len(commandArguments) == 2
            and commandArguments[0] in {"-n", "--lines"}
            and commandArguments[1].isdigit()
        )
    if command == "date":
        return all(token.startswith("+") for token in commandArguments)
    return True


def _is_function_declaration(tokens: list[str]) -> bool:
    """Recognize a function declaration token sequence, which performs no command."""
    return bool(
        tokens
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tokens[0])
        and any(token in {"()", "{"} for token in tokens[1:])
    )


def _has_only_discarding_redirections(tokens: list[str]) -> bool:
    """Reject writes except output discarded into /dev/null or another descriptor."""
    for index, token in enumerate(tokens):
        if token in {">", ">>", "&>"}:
            if index + 1 >= len(tokens) or tokens[index + 1] != "/dev/null":
                return False
        elif token in {"<", "<<"}:
            return False
    return True


def _command_substitutions(tokens: list[str]) -> list[str]:
    """Extract command substitutions embedded in otherwise inert output strings."""
    substitutions = []
    for token in tokens:
        substitutions.extend(re.findall(r"\$\((.*?)\)", token, flags=re.DOTALL))
        substitutions.extend(re.findall(r"`(.*?)`", token, flags=re.DOTALL))
    return substitutions


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


def explain_risky_behavior(script: str) -> dict:
    """Return the clearest available reason for a model-produced risky label."""
    executableText = executable_rule_text(script)
    for reason, pattern, explanation in (*RISKY_RULES, *RISK_EXPLANATION_RULES):
        if pattern.search(executableText):
            return {"reason": reason, "explanation": explanation}
    return {
        "reason": "model_detected_risk",
        "explanation": (
            "the statistical model found risky patterns, but no specific "
            "high-confidence operation was identified"
        ),
    }
