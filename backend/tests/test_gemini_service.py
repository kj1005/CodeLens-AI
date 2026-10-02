import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gemini_service import (
    DEFAULT_EXPLANATION_LEVEL,
    DEFAULT_GEMINI_MODEL,
    EMPTY_CONTEXT_ANSWER,
    ExplanationLevel,
    GeminiConfigurationError,
    GeminiGenerationError,
    GeminiGenerationService,
    build_grounded_prompt,
)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response or SimpleNamespace(text="  Answer from code context.  ")
        self.error = error
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        if self.error is not None:
            raise self.error
        return self.response


class GeminiGenerationServiceTests(unittest.TestCase):
    def test_successful_generation_uses_model_and_returns_clean_answer(self):
        models = FakeModels()
        service = GeminiGenerationService(
            model_name="test-gemini-model",
            client=SimpleNamespace(models=models),
        )

        answer = service.generate_answer(
            "Where is authentication handled?",
            "[File: auth.py | Lines: 1-3 | Language: Python]\ndef authenticate_user(): pass",
        )

        self.assertEqual(answer, "Answer from code context.")
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(models.calls[0]["model"], "test-gemini-model")
        self.assertIn("Explanation level (intermediate)", models.calls[0]["contents"])

    def test_prompt_contains_grounding_rules_question_and_context(self):
        question = "Where does this project connect to the database?"
        context = "[File: database.py | Lines: 4-6 | Language: Python]\nconnect()"

        prompt = build_grounded_prompt(question, context)

        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
        self.assertIn("Do not invent code", prompt)
        self.assertIn("If the context is insufficient", prompt)
        self.assertIn("line ranges", prompt)

    def test_three_levels_have_distinct_instructions_and_same_question_context(self):
        question = "Where is authentication handled?"
        context = "[File: auth.py | Lines: 1-3 | Language: Python]\ndef authenticate_user(): pass"
        prompts = {
            level: build_grounded_prompt(question, context, level)
            for level in ExplanationLevel
        }

        self.assertEqual(len(set(prompts.values())), 3)
        for prompt in prompts.values():
            self.assertIn(question, prompt)
            self.assertIn(context, prompt)
            self.assertIn("Do not invent code", prompt)
            self.assertIn("If the context is insufficient", prompt)
            self.assertIn("file names and line ranges", prompt)

        self.assertIn("limited programming knowledge", prompts[ExplanationLevel.BEGINNER])
        self.assertIn("implementation flow", prompts[ExplanationLevel.INTERMEDIATE])
        self.assertIn("technically detailed", prompts[ExplanationLevel.EXPERT])
        self.assertIn("only when supported", prompts[ExplanationLevel.EXPERT])

    def test_existing_call_defaults_to_intermediate_explanation(self):
        models = FakeModels()
        service = GeminiGenerationService(client=SimpleNamespace(models=models))

        service.generate_answer("Question", "Code context")

        self.assertEqual(DEFAULT_EXPLANATION_LEVEL, ExplanationLevel.INTERMEDIATE)
        self.assertIn("Explanation level (intermediate)", models.calls[0]["contents"])

    def test_invalid_explanation_level_is_rejected_before_api_call(self):
        models = FakeModels()
        service = GeminiGenerationService(client=SimpleNamespace(models=models))

        with self.assertRaises(ValueError):
            service.generate_answer("Question", "Code context", "advanced")

        self.assertEqual(models.calls, [])

    def test_missing_api_key_raises_configuration_error_without_client_call(self):
        with patch.dict(os.environ, {}, clear=True):
            service = GeminiGenerationService()
            with patch("gemini_service.genai.Client") as client_constructor:
                with self.assertRaises(GeminiConfigurationError):
                    service.generate_answer("Question", "Code context")

        client_constructor.assert_not_called()

    def test_api_key_is_read_from_environment_without_exposing_it(self):
        fake_key = "not-a-real-api-key"
        fake_client = SimpleNamespace(models=FakeModels())
        with patch.dict(os.environ, {"GEMINI_API_KEY": fake_key}, clear=True):
            with patch("gemini_service.genai.Client", return_value=fake_client) as client_constructor:
                service = GeminiGenerationService()
                service.generate_answer("Question", "Code context")

        client_constructor.assert_called_once_with(api_key=fake_key)

    def test_model_can_be_configured_through_environment(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "configured-model"}, clear=True):
            service = GeminiGenerationService()

        self.assertEqual(service.model_name, "configured-model")
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-2.5-flash")

    def test_api_error_is_wrapped_without_leaking_error_details(self):
        models = FakeModels(error=RuntimeError("private transport details"))
        service = GeminiGenerationService(client=SimpleNamespace(models=models))

        with self.assertRaisesRegex(GeminiGenerationError, "Gemini answer generation failed") as raised_error:
            service.generate_answer("Question", "Code context")

        self.assertNotIn("private transport details", str(raised_error.exception))

    def test_empty_question_is_rejected_before_calling_client(self):
        models = FakeModels()
        service = GeminiGenerationService(client=SimpleNamespace(models=models))

        with self.assertRaises(ValueError):
            service.generate_answer("   ", "Code context")

        self.assertEqual(models.calls, [])

    def test_empty_context_returns_insufficient_context_answer_without_api_call(self):
        models = FakeModels()
        service = GeminiGenerationService(client=SimpleNamespace(models=models))

        answer = service.generate_answer("Question", "  ")

        self.assertEqual(answer, EMPTY_CONTEXT_ANSWER)
        self.assertEqual(models.calls, [])


if __name__ == "__main__":
    unittest.main()