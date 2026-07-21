# Classifier showcase inputs

These files contain synthetic text inputs for demonstrating the classifier.
They deliberately use the `.bash.txt` extension so they are not mistaken for
scripts that should be run.

The malicious network examples use IANA documentation addresses and the
reserved `.invalid` domain. **Do not execute any of these samples.** Pass them
to the classifier as text with, for example:

```powershell
.\.venv\Scripts\python.exe main.py predict --script-file showcase_scripts\safe_backup.bash.txt
```

Expected primary classifications:

- `safe_backup.bash.txt`: safe
- `safe_health_check.bash.txt`: safe
- `risky_system_cleanup.bash.txt`: risky
- `risky_firewall_change.bash.txt`: risky
- `malicious_reverse_shell.bash.txt`: malicious / reverse shell
- `malicious_exfiltration.bash.txt`: malicious / exfiltration
- `malicious_persistence.bash.txt`: malicious / persistence

These are expected outcomes, not guaranteed ones. Actual predictions depend on
the trained checkpoint and the examples present in the dataset.

