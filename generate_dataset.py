#!/usr/bin/env python3
"""
Generate SAFE and RISKY shell-script datasets for an LLM classifier.

Output schema (JSONL, one record per line) matches the malicious set:
    {"id", "label", "script", "category", "description"}
`category` and `description` are left empty ("") for safe/risky records.

Labeling rubric
---------------
safe   : read-only or user-scoped effects; no root, no destructive ops,
         no fetch-then-execute, no system config changes.
risky  : legitimate admin/install/deploy scripts that do powerful,
         privileged, or irreversible things (sudo, rm -rf, dd, chmod 777,
         curl|bash installers, editing /etc, firewall flush, mkfs, ...).
         Benign intent, dangerous capability.
"""

import argparse
import hashlib
import json
import random

PKGS = ["nginx", "postgresql", "redis-server", "htop", "curl", "git", "vim",
        "docker.io", "ufw", "fail2ban", "build-essential", "python3-pip",
        "nodejs", "jq", "tmux", "net-tools", "unzip", "certbot"]
PY_PKGS = ["requests", "flask", "numpy", "pandas", "django", "fastapi",
           "pytest", "boto3", "sqlalchemy", "pyyaml"]
NPM_PKGS = ["express", "lodash", "react", "typescript", "eslint", "webpack",
            "axios", "jest", "vite", "next"]
SERVICES = ["nginx", "postgresql", "redis", "docker", "myapp", "cron",
            "ssh", "fail2ban", "grafana", "prometheus"]
USER_DIRS = ["~/projects", "~/work", "$HOME/src", "~/backups", "~/data",
             "./build", "./dist", "/tmp/workdir"]
SYS_PATHS = ["/etc/nginx", "/var/www/html", "/opt/app", "/usr/local/bin",
             "/var/lib/myapp", "/srv/data", "/etc/systemd/system"]
REPOS = ["https://github.com/acme/webapp.git",
         "https://github.com/acme/api-service.git",
         "https://github.com/example/tooling.git",
         "git@github.com:org/infra.git"]
VERSIONS = ["1.2.3", "2.0.1", "3.14.0", "0.9.7", "4.5.0", "18.17.1", "20.11.0"]
USERS = ["deploy", "appuser", "svc_web", "ci", "worker"]
PORTS = ["8080", "3000", "5432", "6379", "443", "9090"]

SHEBANGS = ["#!/bin/bash", "#!/usr/bin/env bash", "#!/bin/sh"]
SETLINES = ["set -e", "set -euo pipefail", "set -eu", ""]

R = random.Random()  # seeded in main()


def pick(seq):
    return R.choice(seq)


def header(comment):
    """Randomized shebang + strictness line + comment -> entropy multiplier."""
    sb = pick(SHEBANGS)
    st = pick(SETLINES)
    lines = [sb]
    if st:
        lines.append(st)
    lines.append(f"# {comment}")
    return "\n".join(lines) + "\n"



def safe_greet():
    name = pick(["World", "team", "developer", "user"])
    return f"""#!/bin/bash
# print a friendly greeting
echo "Hello, {name}!"
echo "Today is $(date '+%Y-%m-%d')"
"""


def safe_list_dir():
    d = pick(USER_DIRS)
    return f"""#!/bin/bash
set -euo pipefail
# list files in a project directory, newest first
target="{d}"
if [ -d "$target" ]; then
    ls -lah --time-style=long-iso "$target" | sort -k6
else
    echo "Directory $target not found" >&2
fi
"""


def safe_git_clone():
    repo = pick(REPOS)
    d = pick(USER_DIRS)
    return f"""#!/bin/bash
set -e
# clone or update a repository into the workspace
dest="{d}/$(basename "{repo}" .git)"
if [ -d "$dest/.git" ]; then
    git -C "$dest" pull --ff-only
else
    git clone {repo} "$dest"
fi
"""


def safe_py_venv():
    pkgs = " ".join(R.sample(PY_PKGS, k=R.randint(1, 3)))
    return f"""#!/bin/bash
set -euo pipefail
# create an isolated virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install {pkgs}
echo "Environment ready."
"""


def safe_npm_build():
    return f"""#!/bin/bash
set -e
# install node dependencies and run the build (no global installs)
cd "$(dirname "$0")"
npm ci
npm run build
echo "Build complete -> ./dist"
"""


def safe_make_build():
    return f"""#!/bin/bash
set -e
# configure and build from source into a local prefix
./configure --prefix="$HOME/.local"
make -j"$(nproc)"
make check || true
"""


def safe_backup_home():
    d = pick(["~/projects", "~/documents", "~/work"])
    return f"""#!/bin/bash
set -euo pipefail
# archive a user directory into the backups folder
src="{d}"
stamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HOME/backups"
tar -czf "$HOME/backups/backup-$stamp.tar.gz" -C "$(dirname "$src")" "$(basename "$src")"
echo "Backup written to $HOME/backups/backup-$stamp.tar.gz"
"""


def safe_log_scan():
    pat = pick(["ERROR", "WARN", "timeout", "failed login", "404"])
    return f"""#!/bin/bash
set -euo pipefail
# count occurrences of a pattern in application logs
logfile="${{1:-./app.log}}"
if [ ! -f "$logfile" ]; then
    echo "No log file at $logfile" >&2
    exit 0
fi
count=$(grep -c "{pat}" "$logfile" || true)
echo "Found $count lines matching '{pat}'"
"""


def safe_env_report():
    return """#!/bin/bash
# print a small diagnostic report about the environment
echo "User:    $(whoami)"
echo "Host:    $(hostname)"
echo "Shell:   $SHELL"
echo "Uptime:  $(uptime -p 2>/dev/null || uptime)"
echo "Disk:"
df -h --output=source,size,used,avail / 2>/dev/null || df -h /
"""


def safe_health_check():
    port = pick(PORTS)
    return f"""#!/bin/bash
set -euo pipefail
# probe a local service and report status
url="http://127.0.0.1:{port}/health"
if curl -fsS --max-time 5 "$url" >/dev/null; then
    echo "Service on port {port} is healthy"
else
    echo "Service on port {port} is not responding" >&2
    exit 1
fi
"""


def safe_csv_summary():
    return """#!/bin/bash
set -euo pipefail
# summarize a CSV: row count and column count
file="${1:?usage: summarize.sh <file.csv>}"
rows=$(($(wc -l < "$file") - 1))
cols=$(head -1 "$file" | awk -F',' '{print NF}')
echo "Rows: $rows, Columns: $cols"
"""


def safe_rename_batch():
    ext = pick(["txt", "log", "md", "csv"])
    return f"""#!/bin/bash
set -euo pipefail
# lowercase all .{ext} filenames in the current directory
for f in ./*.{ext}; do
    [ -e "$f" ] || continue
    lower="$(echo "$f" | tr 'A-Z' 'a-z')"
    [ "$f" != "$lower" ] && mv -n "$f" "$lower"
done
"""


def safe_wait_for_port():
    port = pick(PORTS)
    return f"""#!/bin/bash
set -euo pipefail
# wait until a TCP port accepts connections (used before running tests)
port={port}
for i in $(seq 1 30); do
    if (echo > /dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
        echo "Port $port is up"
        exit 0
    fi
    sleep 1
done
echo "Timed out waiting for port $port" >&2
exit 1
"""


def safe_json_pretty():
    return """#!/bin/bash
set -euo pipefail
# pretty-print and validate a JSON file
file="${1:?usage: fmt.sh <file.json>}"
if jq . "$file" > "$file.tmp"; then
    mv "$file.tmp" "$file"
    echo "Formatted $file"
else
    rm -f "$file.tmp"
    echo "Invalid JSON in $file" >&2
    exit 1
fi
"""


def safe_disk_usage():
    d = pick(USER_DIRS)
    msg = pick(["disk usage", "space report", "size breakdown"])
    return header(f"show {msg} for a project directory") + f"""du -sh {d}/* 2>/dev/null | sort -rh | head -n {R.randint(5, 20)}
"""


def safe_find_large():
    d = pick(USER_DIRS)
    mb = pick([50, 100, 200, 500])
    return header(f"list files larger than {mb}MB (read-only)") + f"""find {d} -type f -size +{mb}M -exec ls -lh {{}} \\; 2>/dev/null | awk '{{print $5, $9}}'
"""


def safe_checksum():
    algo = pick(["sha256sum", "sha1sum", "md5sum"])
    return header("verify the checksum of a downloaded file") + f"""file="${{1:?usage: verify.sh <file>}}"
{algo} "$file"
"""


def safe_download_only():
    d = pick(["~/Downloads", "./downloads", "$HOME/data"])
    return header("download a file to disk (no execution)") + f"""url="${{1:?usage: fetch.sh <url>}}"
mkdir -p {d}
curl -fsSL -o "{d}/$(basename "$url")" "$url"
echo "Saved to {d}"
"""


def safe_ping_check():
    host = pick(["8.8.8.8", "example.com", "github.com", "1.1.1.1"])
    n = pick([3, 4, 5])
    return header("check network connectivity") + f"""if ping -c {n} {host} >/dev/null 2>&1; then
    echo "{host} is reachable"
else
    echo "{host} is unreachable" >&2
fi
"""


def safe_tar_extract():
    d = pick(USER_DIRS)
    return header("extract an archive into the workspace") + f"""archive="${{1:?usage: extract.sh <archive.tar.gz>}}"
mkdir -p {d}
tar -xzf "$archive" -C {d}
echo "Extracted into {d}"
"""


def safe_grep_todo():
    tag = pick(["TODO", "FIXME", "XXX", "HACK"])
    d = pick(USER_DIRS)
    return header(f"list {tag} comments across the codebase") + f"""grep -rn "{tag}" {d} --include='*.py' --include='*.js' 2>/dev/null || echo "No {tag} found"
"""


def safe_count_files():
    d = pick(USER_DIRS)
    ext = pick(["py", "js", "go", "sh", "md"])
    return header("count source files by type") + f"""n=$(find {d} -type f -name '*.{ext}' 2>/dev/null | wc -l)
echo "Found $n .{ext} files"
"""


def safe_uptime_log():
    return header("append an uptime sample to a local log") + f"""log="$HOME/uptime.log"
echo "$(date '+%F %T') $(uptime | sed 's/.*load average://')" >> "$log"
tail -n {R.randint(3, 10)} "$log"
"""


def safe_git_status():
    d = pick(USER_DIRS)
    return header("report git status for a repo") + f"""repo="{d}"
if [ -d "$repo/.git" ]; then
    git -C "$repo" status --short --branch
else
    echo "$repo is not a git repository" >&2
fi
"""


def safe_lint():
    tool = pick(["ruff check", "eslint", "shellcheck", "flake8", "black --check"])
    d = pick(USER_DIRS)
    return header("run a linter (read-only, no changes)") + f"""{tool} {d} || echo "Lint issues found"
"""


def safe_gen_password():
    n = pick([16, 20, 24, 32])
    return header("generate a random password locally and print it") + f"""LC_ALL=C tr -dc 'A-Za-z0-9!@#%' </dev/urandom | head -c {n}; echo
"""


def safe_watch_file():
    return header("print the tail of a log and follow it briefly") + f"""log="${{1:-./app.log}}"
[ -f "$log" ] && tail -n {R.randint(10, 50)} "$log" || echo "No log at $log"
"""


def safe_json_field():
    field = pick([".name", ".version", ".dependencies", ".scripts.build", ".id"])
    return header("extract a field from a JSON file") + f"""file="${{1:?usage: get.sh <file.json>}}"
jq -r '{field}' "$file"
"""


SAFE_TEMPLATES = [
    safe_greet, safe_list_dir, safe_git_clone, safe_py_venv, safe_npm_build,
    safe_make_build, safe_backup_home, safe_log_scan, safe_env_report,
    safe_health_check, safe_csv_summary, safe_rename_batch,
    safe_wait_for_port, safe_json_pretty,
    safe_disk_usage, safe_find_large, safe_checksum, safe_download_only,
    safe_ping_check, safe_tar_extract, safe_grep_todo, safe_count_files,
    safe_uptime_log, safe_git_status, safe_lint, safe_gen_password,
    safe_watch_file, safe_json_field,
]


# ----------------------------------------------------------------------------
# RISKY templates (legit admin/install/deploy, but powerful/privileged)
# ----------------------------------------------------------------------------
def risky_apt_install():
    pkgs = " ".join(R.sample(PKGS, k=R.randint(2, 4)))
    return f"""#!/bin/bash
set -e
# system package installation (requires root)
sudo apt-get update
sudo apt-get install -y {pkgs}
sudo apt-get clean
"""


def risky_curl_bash():
    return f"""#!/bin/bash
set -e
# vendor install script: fetch and execute installer as root
curl -fsSL https://get.example.com/install.sh | sudo bash
"""


def risky_add_apt_repo():
    return f"""#!/bin/bash
set -e
# add third-party apt repository and signing key
curl -fsSL https://repo.example.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/example.gpg
echo "deb [signed-by=/usr/share/keyrings/example.gpg] https://repo.example.com stable main" \\
    | sudo tee /etc/apt/sources.list.d/example.list
sudo apt-get update
"""


def risky_systemd_unit():
    svc = pick(SERVICES)
    path = pick(SYS_PATHS)
    return f"""#!/bin/bash
set -e
# install and enable a systemd service (root required)
sudo tee /etc/systemd/system/{svc}.service > /dev/null <<'EOF'
[Unit]
Description={svc} service
After=network.target

[Service]
ExecStart={path}/{svc} --serve
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now {svc}
"""


def risky_rm_cleanup():
    path = pick(["/var/log", "/tmp", "/opt/app/cache", "/var/cache/myapp"])
    return f"""#!/bin/bash
set -e
# purge a directory to reclaim disk (destructive, irreversible)
target="{path}"
echo "Removing everything under $target"
sudo rm -rf "$target"/*
"""


def risky_chmod_777():
    path = pick(SYS_PATHS)
    return f"""#!/bin/bash
set -e
# open up permissions so the web server can write (overly broad)
sudo chmod -R 777 {path}
sudo chown -R www-data:www-data {path}
"""


def risky_dd_image():
    return f"""#!/bin/bash
set -e
# write a disk image to a block device (DESTROYS existing data)
img="${{1:?usage: flash.sh <image>}}"
dev="${{2:?usage: flash.sh <image> <device e.g. /dev/sdb>}}"
sudo dd if="$img" of="$dev" bs=4M status=progress conv=fsync
sync
"""


def risky_mkfs():
    dev = pick(["/dev/sdb1", "/dev/nvme0n1p1", "/dev/vdb1"])
    return f"""#!/bin/bash
set -e
# format and mount a new data volume (erases the partition)
sudo mkfs.ext4 -F {dev}
sudo mkdir -p /mnt/data
sudo mount {dev} /mnt/data
echo "{dev} /mnt/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
"""


def risky_iptables_flush():
    port = pick(PORTS)
    return f"""#!/bin/bash
set -e
# reset firewall rules and allow a service port (drops all existing rules)
sudo iptables -F
sudo iptables -P INPUT ACCEPT
sudo iptables -A INPUT -p tcp --dport {port} -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || true
"""


def risky_ufw_disable():
    return """#!/bin/bash
set -e
# temporarily disable the host firewall for troubleshooting
sudo ufw disable
sudo systemctl stop fail2ban 2>/dev/null || true
echo "Firewall disabled - remember to re-enable it"
"""


def risky_add_user():
    u = pick(USERS)
    return f"""#!/bin/bash
set -e
# create a service account with sudo and passwordless login
sudo useradd -m -s /bin/bash {u}
sudo usermod -aG sudo {u}
echo "{u} ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/{u}
sudo chmod 440 /etc/sudoers.d/{u}
"""


def risky_docker_prune():
    return """#!/bin/bash
set -e
# reclaim docker disk: removes ALL unused images, containers, volumes
docker system prune -af --volumes
docker builder prune -af
"""


def risky_docker_privileged():
    port = pick(PORTS)
    return f"""#!/bin/bash
set -e
# run a container with host access (privileged, host network)
docker run -d --privileged --network host \\
    -v /:/host \\
    -p {port}:{port} \\
    --restart always \\
    example/agent:latest
"""


def risky_crontab_install():
    path = pick(SYS_PATHS)
    return f"""#!/bin/bash
set -e
# install a root cron job for periodic maintenance
( sudo crontab -l 2>/dev/null; echo "0 3 * * * {path}/maintenance.sh" ) \\
    | sudo crontab -
"""


def risky_swap_setup():
    return """#!/bin/bash
set -e
# create and enable a swap file (system-level change)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
"""


def risky_sysctl_tune():
    return """#!/bin/bash
set -e
# apply kernel network tuning persistently
sudo tee /etc/sysctl.d/99-tuning.conf > /dev/null <<'EOF'
net.core.somaxconn = 65535
net.ipv4.ip_forward = 1
vm.swappiness = 10
EOF
sudo sysctl --system
"""


def risky_deploy_release():
    path = pick(SYS_PATHS)
    svc = pick(SERVICES)
    return f"""#!/bin/bash
set -e
# deploy a release: overwrite app dir and restart service (root)
sudo systemctl stop {svc}
sudo rm -rf {path}/current
sudo cp -r ./build {path}/current
sudo chown -R {pick(USERS)}:www-data {path}/current
sudo systemctl start {svc}
"""


def risky_disable_selinux():
    return """#!/bin/bash
set -e
# set SELinux to permissive (weakens host security)
sudo setenforce 0
sudo sed -i 's/^SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config
"""


RISKY_TEMPLATES = [
    risky_apt_install, risky_curl_bash, risky_add_apt_repo, risky_systemd_unit,
    risky_rm_cleanup, risky_chmod_777, risky_dd_image, risky_mkfs,
    risky_iptables_flush, risky_ufw_disable, risky_add_user, risky_docker_prune,
    risky_docker_privileged, risky_crontab_install, risky_swap_setup,
    risky_sysctl_tune, risky_deploy_release, risky_disable_selinux,
]


def strip_comments(script):
    """Drop all comment lines (keep the shebang) so the model learns on code
    only, not on comments that narrate the label."""
    out = []
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("#") and not s.startswith("#!"):
            continue
        out.append(line)
    return "\n".join(out).strip("\n") + "\n"


def generate(label, templates, count):
    seen = set()
    records = []
    attempts = 0
    max_attempts = count * 200
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        script = strip_comments(pick(templates)())
        h = hashlib.sha1(script.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        records.append({
            "id": f"{label}-{len(records) + 1:05d}",
            "label": label,
            "script": script,
            "category": "",
            "description": "",
        })
    if len(records) < count:
        print(f"WARNING: only produced {len(records)}/{count} unique "
              f"{label} scripts (template space exhausted). Add more templates.")
    return records


def write_jsonl(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {path}")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_malicious_jsonl(path):
    """Load malicious samples and normalize them to the training schema."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            records.append({
                "id": source.get("id", f"malicious-{len(records) + 1:05d}"),
                "label": "malicious",
                "script": source["script"],
                "category": source.get("category", ""),
                "description": source.get("description", ""),
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1500,
                    help="records per class (default 1500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--malicious", default="data/malicious.jsonl",
                    help="malicious JSONL to normalize and include in combined output")
    ap.add_argument("--combine-only", action="store_true",
                    help="keep existing safe/risky files and only rebuild malicious/combined")
    args = ap.parse_args()

    R.seed(args.seed)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    safe_path = os.path.join(args.outdir, "safe.jsonl")
    risky_path = os.path.join(args.outdir, "risky.jsonl")
    if args.combine_only:
        safe = read_jsonl(safe_path)
        risky = read_jsonl(risky_path)
    else:
        safe = generate("safe", SAFE_TEMPLATES, args.count)
        risky = generate("risky", RISKY_TEMPLATES, args.count)
        write_jsonl(safe_path, safe)
        write_jsonl(risky_path, risky)

    malicious = read_malicious_jsonl(args.malicious)
    write_jsonl(args.malicious, malicious)

    # Emit all three labels in one shuffled file for training/testing.
    combined = safe + risky + malicious
    R.shuffle(combined)
    write_jsonl(os.path.join(args.outdir, "combined.jsonl"), combined)


if __name__ == "__main__":
    main()
