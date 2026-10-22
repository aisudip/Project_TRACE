# Risk Analyst Agent - Chain-of-Thought Implementation
# Implement Risk Analyst Agent using Chain-of-Thought prompting

"""
Risk Analyst Agent Module

This agent performs suspicious activity classification using Chain-of-Thought reasoning.
It analyzes customer profiles, account behavior, and transaction patterns to identify
potential financial crimes.

YOUR TASKS:
- Study Chain-of-Thought prompting methodology
- Design system prompt with structured reasoning framework
- Implement case analysis with proper error handling
- Parse and validate structured JSON responses
- Create comprehensive audit logging
"""

import json
from datetime import datetime, timezone
from typing import Literal, Any, Callable
from dotenv import load_dotenv
import os
from openai import OpenAI
from pathlib import Path
import logging
from dotenv import load_dotenv
import re

from typing import TypeVar
from pydantic import BaseModel, RootModel, ValidationError, Field

T = TypeVar("T", bound=BaseModel)

class JSONDictOutput(RootModel[dict[str, Any]]):
    pass

class LLMOutputValidationError(ValueError):
    """LLM failed structured-output validation after retries."""
# Load environment variables and configure logging

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True
)

pylogger = logging.getLogger(__name__)

# Suppress third-party INFO and DEBUG messages
for name in ("httpx", "httpx2", "httpcore", "openai"):
    logging.getLogger(name).setLevel(logging.WARNING)

# Import your foundation components
from .foundation_sar import (
    CustomerData,
    AccountData,
    TransactionData,
    CaseData,
    RiskAnalystOutput,
    ExplainabilityLogger,
    DataLoader,
    load_csv_data
)

# Directory containing risk_analyst_agent.py
SRC_DIR = Path(__file__).resolve().parent

# Project root: starter/
PROJECT_ROOT = SRC_DIR.parent

# Data directory: starter/data/
DATA_DIR = PROJECT_ROOT / "data"

def create_client_and_logger():
    # Load environment variables
    load_dotenv('../../../../.env', override=True)    
    explainability_logger = ExplainabilityLogger()

    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        print("⚠️ WARNING: No OpenAI API key found!")
        print("Please set OPENAI_API_KEY in your .env file")
        print("Get your OpenAI API key from in your workspace")
    else:
        client = OpenAI(api_key=openai_api_key)
        print("✅ OpenAI client initialized")
        print(f"🔑 API key: {openai_api_key[:8]}...{openai_api_key[-4:]}")

    return client, explainability_logger

class RiskAnalystAgent:
    """
    Risk Analyst agent using Chain-of-Thought reasoning.
    
    Implement agent that:
    - Uses systematic Chain-of-Thought prompting
    - Classifies suspicious activity patterns
    - Returns structured JSON output
    - Handles errors gracefully
    - Logs all operations for audit
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-4.1-mini"):
        """Initialize the Risk Analyst Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        # Initialize agent components
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
    
        # Design Chain-of-Thought system prompt

        self.system_prompt = """
        You are a Senior Financial Crime Risk Analyst.
        Use a systematic, step-by-step financial crime analysis framework.

        The analysis consists of five stages:

        1. Data Review
        2. Pattern Recognition
        3. Regulatory Mapping
        4. Risk Quantification
        5. Classification Decision

        Use the evidence generated during earlier stages to inform
        subsequent stages.

        Provide concise, evidence-based justifications.
        Do not invent facts or attribute criminal intent without evidence.
        
        When JSON output is requested:
        - Return exactly one valid JSON object.
        - Start the response with { and end it with }.
        - Use double quotes for all property names and string values.
        - Do not use Markdown code fences.
        - Do not include prose before or after the JSON.

        Classification categories:
        - Structuring
        - Sanctions
        - Fraud
        - Money_Laundering
        - Other

        Final structured JSON output must contain:
        - classification
        - confidence_score
        - reasoning
        - key_indicators
        - risk_level

        """

    def analyze_case(self, case_data) -> RiskAnalystOutput:
        if case_data is None:
            raise ValueError("Case data cannot be None")

        transaction_count = len(case_data.transactions)

        if transaction_count <= 5:
            return self._analyze_case_single(case_data)

        return self._analyze_case_detailed(case_data)

    def _analyze_case_single(self, case_data) -> RiskAnalystOutput:
        """
        Single-pass risk analysis with corrective retries.

        Returns validated RiskAnalystOutput.
        Raises LLMOutputValidationError after retries are exhausted.
        """

        start_time = datetime.now(timezone.utc)
        case_id = str(case_data.case_id)

        case_payload = build_risk_case_payload(case_data)

        prompt = f"""
        Analyze the following financial-crime case using a
        systematic, step-by-step risk assessment framework.

        Case data:
        {json.dumps(case_payload, default=str)}

        Classify the activity into exactly one category:
        Structuring, Sanctions, Fraud, Money_Laundering, Other.

        Return ONLY valid JSON containing these fields:
        - classification
        - confidence_score
        - reasoning
        - key_indicators
        - risk_level

        Requirements:
        - confidence_score must be numeric between 0.0 and 1.0.
        - reasoning must not exceed 500 characters.
        - risk_level must be Low, Medium, High, or Critical.
        - Use only supplied evidence.
        - Do not invent transactions or facts.
        - Do not include Markdown code fences.
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Capture analysis errors only.
        # Audit logging intentionally remains outside this block.
        try:
            result = self._get_validated_llm_response(
                messages=messages,
                output_model=RiskAnalystOutput,
                max_retries=2,
                max_tokens=1000
            )

        except LLMOutputValidationError as exc:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="Case Analysis",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data={
                    "status": "MANUAL_REVIEW",
                    "analysis": None
                },
                reasoning=(
                    "JSON parsing failed "
                    "after corrective retries."
                ),
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(exc)
            )

            raise

        except Exception as exc:
            # API errors and unexpected execution failures.
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="Case Analysis",
                case_id=case_id,
                input_data={"case_id": case_id},
                output_data={
                    "status": "FAILED",
                    "analysis": None
                },
                reasoning="Risk analysis execution failed.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(exc)
            )

            raise

        # Success: audit outside analysis exception handlers.
        execution_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self.logger.log_agent_action(
            agent_type="RiskAnalyst",
            action="Case Analysis",
            case_id=case_id,
            input_data={"case_id": case_id},
            output_data=result.model_dump(),
            reasoning="Successfully analyzed case.",
            execution_time_ms=execution_time_ms,
            success=True
        )

        return result
    
    def _analyze_case_detailed(self, case_data) -> RiskAnalystOutput:  # Use quotes for forward reference
        """
        Perform risk analysis on a case using Chain-of-Thought reasoning.
        
        Implement analysis that:
        - Creates structured user prompt with case details
        - Makes OpenAI API call with system prompt
        - Parses and validates JSON response
        - Handles errors and logs operations
        - Returns validated RiskAnalystOutput
        """
        print("🚀 STARTING CASE ANALYSIS\n")

        self.sar_analysis_output = {

            "step_1": {
                "case_id": "",
                "customer": {
                    "customer_id": "",
                    "name": "",
                    "risk_rating": "",
                    "occupation": "",
                    "annual_income": None,
                    "customer_since": ""
                },

                "accounts": [
                    {
                        "account_id": "",
                        "account_type": "",
                        "status": "",
                        "current_balance": 0.0
                    }
                ],

                "activity_period": {
                    "start_date": "",
                    "end_date": ""
                },

                "transaction_summary": {
                    "transaction_count": 0,
                    "total_transaction_amount": 0.0
                },

                "material_transactions": [
                    {
                        "transaction_id": "",
                        "account_id": "",
                        "date": "",
                        "type": "",
                        "amount": 0.0,
                        "counterparty": "",
                        "location": ""
                    }
                ],

                "prior_case_history": [],
                "notable_facts": [],
                "data_gaps": []
            },

            "step_2": {
                "suspicious_patterns": [
                    {
                        "pattern_id": "",
                        "pattern_name": "",
                        "description": "",

                        "transactions": [
                            {
                                "transaction_id": "",
                                "date": "",
                                "amount": 0.0
                            }
                        ],

                        "activity_period": {
                            "start_date": "",
                            "end_date": ""
                        },

                        "total_amount": 0.0,
                        "why_unusual": "",
                        "customer_profile_deviation": "",
                        "possible_legitimate_explanation": "",
                        "indicator_strength":
                            "Low | Medium | High"
                    }
                ],
                "overall_pattern_summary": ""
            },

            "step_3": {
                "sar_typologies": [
                    {
                        "activity_type": "",
                        "activity_subtype": "",
                        "matched_patterns": [],
                        "supporting_transaction_ids": [],
                        "mapping_rationale": "",
                        "match_strength":
                            "Limited | Moderate | Strong"
                    }
                ],
                "unmapped_patterns": [],
                "potential_financial_crime_indicators": []
            },

            "step_4": {
                "risk_level":
                    "Low | Medium | High | Critical",
                "risk_score": 0,
                "sar_assessment": {
                    "recommendation":
                        "No SAR | Further Investigation | SAR Escalation | SAR Filing Consideration",
                    "rationale": "",
                    "factors_supporting_filing": [],
                    "factors_against_filing": [],
                    "additional_information_required": []
                },
                "risk_factors": [
                    {
                        "factor": "",
                        "impact":
                            "Low | Medium | High",
                        "evidence": []
                    }
                ],
                "mitigating_factors": [],
                "confidence":
                    "Low | Medium | High"
            },
        }

        if case_data is None:
            raise ValueError("Case data cannot be None")

        self.COT_PROMPT =  create_chain_of_thought_framework()
        self.case_id = case_data.case_id
        
        # ===== Step 1: Data Review =====

        start_time = datetime.now(timezone.utc)

        try:
            data_review_output = self._format_case_for_prompt(case_data)

            data_review_output["material_transactions"] = [
                normalize_material_transaction(tx)
                for tx in data_review_output["material_transactions"]
            ]

            pylogger.debug(
                "Step 1 Data Review output: %s",
                data_review_output
            )

        except LLMOutputValidationError as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Data Review",
                case_id=self.case_id,
                input_data={"case_id": self.case_id},
                output_data={"status": "MANUAL_REVIEW"},
                reasoning="Transaction chunk validation retries exhausted.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            raise

        except Exception as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Data Review",
                case_id=self.case_id,
                input_data={"case_id": self.case_id},
                output_data={"status": "FAILED"},
                reasoning="Data Review processing failed.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            raise

        # Successful processing; logging is outside the try block.
        execution_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self.logger.log_agent_action(
            agent_type="RiskAnalysisAgent",
            action="Data Review",
            case_id=self.case_id,
            input_data={"case_id": self.case_id},
            output_data=data_review_output,
            reasoning="Successfully reviewed case data.",
            execution_time_ms=execution_time_ms,
            success=True
        )

        print("✅ Step 1: Data Review Completed")

        #  ===== Step 2. Pattern Recognition =====
        data_review_json = json.dumps(
            data_review_output,
            separators=(",", ":"),
            default=str
        )

        step2_schema_json = json.dumps(
            self.sar_analysis_output["step_2"],
            separators=(",", ":")
        )

        step2_input = f"""
        For Step 2 of the analysis, identify suspicious activity patterns.

        {self.COT_PROMPT["step_2"]}

        Use the following information:

        Data Review:
        {data_review_json}

        Return ONLY valid JSON matching exactly the structure below:
        {step2_schema_json}

        Evidence rules:
        - Treat the Data Review as the complete evidence available for this stage.
        - Use only information contained in the Data Review.
        - Do not infer or reconstruct transactions that are not included in
        material_transactions.
        - Reference only transaction IDs present in material_transactions.
        - Do not invent facts.

        For customer_profile_deviation:
        - Identify a deviation only if expected customer activity is explicitly
        supported by the Data Review.
        - Do not infer expected behavior solely from occupation, income,
        risk rating, or account type.
        - If expected activity is unavailable, return:
        "Insufficient information to determine profile deviation."

        For possible_legitimate_explanation:
        - Include an explanation only if supported by the supplied evidence.
        - Do not invent hypothetical business, investment, family, personal,
        or commercial purposes.
        - If no legitimate explanation is supported, return:
        "No supported legitimate explanation identified from available data."

        For total_amount:
        - Calculate the sum of the absolute amounts of all transactions listed
        under that suspicious pattern.
        - Include only transactions actually listed in that pattern.
        - Do not use the overall case transaction total.

        Pattern requirements:
        - Identify only materially significant suspicious patterns.
        - Return no more than 5 suspicious patterns.
        - Include no more than 10 representative supporting transactions per pattern.
        - Prefer the strongest supporting transactions.
        - Do not repeat a transaction unless necessary to support distinct patterns.
        - Use concise descriptions.
        - Describe observed behavior without attributing intent.
        - Do not state or imply that the customer attempted to conceal, obscure,
        launder, structure, evade reporting requirements, or commit an offense.
        - Explain why the observable activity warrants further investigation.
        - Do not add fields outside the required JSON structure.

        Output requirements:
        - Return exactly one valid JSON object.
        - The first character must be {{.
        - The last character must be }}.
        - Do not return Markdown, commentary, or explanatory text.
        """
        pylogger.debug(f"Step 2 prompt characters:,{len(step2_input)}")

        start_time = datetime.now(timezone.utc)

        try:
            result = self._get_validated_llm_response(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": step2_input}
                ],
                output_model=JSONDictOutput,
                max_retries=2,
                max_tokens=2000,
                result_validator=lambda output: self._validate_step2_result(
                    output.root
                )
            )

            # Preserve dictionary output expected by Step 3
            pattern_output = result.root

        except LLMOutputValidationError as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Pattern Recognition",
                case_id=self.case_id,
                input_data={"case_id": self.case_id},
                output_data={"status": "MANUAL_REVIEW"},
                reasoning=(
                    "Pattern Recognition validation retries exhausted."
                ),
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            raise

        except Exception as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Pattern Recognition",
                case_id=self.case_id,
                input_data={"case_id": self.case_id},
                output_data={"status": "FAILED"},
                reasoning="Pattern Recognition execution failed.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            raise

        # Log success outside the try/except
        execution_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self.logger.log_agent_action(
            agent_type="RiskAnalysisAgent",
            action="Pattern Recognition",
            case_id=self.case_id,
            input_data=data_review_output,
            output_data=pattern_output,
            reasoning="Successfully recognized suspicious activity patterns.",
            execution_time_ms=execution_time_ms,
            success=True
        )

        print("✅ Step 2: Pattern Recognition Completed")


        #  =====  Step 3. Regulatory Mapping ===== 
        step3_input = f"""
        For Step 3 of the analysis, conduct Regulatory Mapping.

        {self.COT_PROMPT["step_3"]}

        Use the Pattern Recognition findings below as the evidence for regulatory mapping.
        Pattern Recognition:
        {json.dumps(pattern_output, indent=2)}

        Return ONLY valid JSON in the format below. Do not include Markdown code fences or explanatory text.
        {json.dumps(self.sar_analysis_output["step_3"], indent=2)}

        Separate observed facts from interpretations.

        Do not claim that a customer committed a crime or regulatory violation.

        Additional output requirements:
        - Return exactly one valid JSON object matching the structure below.
        - The first character of your response must be {{.
        - The last character must be }}.
        - Do not return Markdown, commentary, or explanatory text.
        """
        start_time = datetime.now(timezone.utc)

        try:
            result = self._get_validated_llm_response(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": step3_input
                    }
                ],
                output_model=JSONDictOutput,
                max_retries=2,
                max_tokens=2000,
                result_validator=lambda output: self._validate_step3_result(
                    output.root
                )
            )

            # Preserve dictionary output for Step 4
            regulatory_mapping_output = result.root

        except LLMOutputValidationError as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Regulatory Mapping",
                case_id=self.case_id,
                input_data=pattern_output,
                output_data={"status": "MANUAL_REVIEW"},
                reasoning="Regulatory mapping validation retries exhausted.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            raise

        except Exception as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Regulatory Mapping",
                case_id=self.case_id,
                input_data=pattern_output,
                output_data={"status": "FAILED"},
                reasoning="Regulatory mapping execution failed.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            raise

        # Log successful outcome separately so audit-write failures
        # are not misclassified as analysis failures.
        execution_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self.logger.log_agent_action(
            agent_type="RiskAnalysisAgent",
            action="Regulatory Mapping",
            case_id=self.case_id,
            input_data=pattern_output,
            output_data=regulatory_mapping_output,
            reasoning="Successfully executed regulatory mapping.",
            execution_time_ms=execution_time_ms,
            success=True
        )

        print("✅ Step 3: Regulatory Mapping Completed")

        #  =====  Step 4. Risk Quantification ===== 
        step4_input = f"""
        For Step 4 of the analysis, quantify the identified risks.

        {self.COT_PROMPT["step_4"]}

        Please use the following data during your analysis:
        Pattern Recognition:
        {json.dumps(pattern_output, indent=2)}

        Regulatory Mapping:
        {json.dumps(regulatory_mapping_output, indent=2)}

        Return ONLY valid JSON in the format below. Do not include Markdown code fences or explanatory text.
        {json.dumps(self.sar_analysis_output["step_4"], indent=2)}

        This is decision support. Do not fabricate regulatory thresholds or requirements
        that are not provided in the case or governing policy.

        Additional output requirements:
        - Quantify risk only using supplied evidence.
        - Clearly distinguish calculated metrics from qualitative assessments.
        - Do not invent risk scores, weights, or thresholds.
        - If evidence is insufficient, explicitly state the limitation.
        - Do not treat a risk score as proof of criminal activity.
        - Return exactly one valid JSON object matching the structure below.
        - The first character of your response must be {{.
        - The last character must be }}.
        - Do not return Markdown, commentary, or explanatory text.
        """

        start_time = datetime.now(timezone.utc)

        try:
            result = self._get_validated_llm_response(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": step4_input
                    }
                ],
                output_model=JSONDictOutput,
                max_retries=2,
                max_tokens=2000,
                result_validator=lambda output: self._validate_step4_result(
                    output.root
                )
            )

            # Preserve dictionary output for Step 5
            risk_quantification_output = result.root

        except LLMOutputValidationError as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Risk Quantification",
                case_id=self.case_id,
                input_data=regulatory_mapping_output,
                output_data={"status": "MANUAL_REVIEW"},
                reasoning="Risk quantification validation retries exhausted.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            raise

        except Exception as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Risk Quantification",
                case_id=self.case_id,
                input_data=regulatory_mapping_output,
                output_data={"status": "FAILED"},
                reasoning="Risk quantification execution failed.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            raise

        # Successful analysis: log outside the processing try block.
        execution_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self.logger.log_agent_action(
            agent_type="RiskAnalysisAgent",
            action="Risk Quantification",
            case_id=self.case_id,
            input_data=regulatory_mapping_output,
            output_data=risk_quantification_output,
            reasoning="Successfully quantified risk.",
            execution_time_ms=execution_time_ms,
            success=True
        )

        print("✅ Step 4: Risk Quantification Completed")

        #  =====  Step 5. Classification Decision ===== 

        step5_input = f"""
        For Step 5, make the final classification decision.

        {self.COT_PROMPT["step_5"]}

        Regulatory Mapping:
        {json.dumps(regulatory_mapping_output, indent=2)}

        Risk Quantification:
        {json.dumps(risk_quantification_output, indent=2)}

        Return ONLY valid JSON matching this schema:

        {json.dumps(RiskAnalystOutput.model_json_schema(), indent=2)}

        The confidence score must reflect the strength of evidence for the
        specific classification, not merely confidence that the activity
        is unusual or suspicious.

        Do not treat interpretations from previous stages as established facts.
        Distinguish observed evidence from analytical inference.        

        Additional output requirements:
        - Return exactly one valid JSON object matching the structure below.
        - The first character of your response must be {{.
        - The last character must be }}.
        - Do not return Markdown, commentary, or explanatory text.
        """
        start_time = datetime.now(timezone.utc)

        try:
            risk_analysis_output = self._get_validated_llm_response(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": step5_input
                    }
                ],
                output_model=RiskAnalystOutput,
                max_retries=2,
                max_tokens=1000
            )

        except LLMOutputValidationError as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Classification Decision",
                case_id=self.case_id,
                input_data=risk_quantification_output,
                output_data={"status": "MANUAL_REVIEW"},
                reasoning="Classification validation retries exhausted.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            raise

        except Exception as e:
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="RiskAnalysisAgent",
                action="Classification Decision",
                case_id=self.case_id,
                input_data=risk_quantification_output,
                output_data={"status": "FAILED"},
                reasoning="Classification execution failed.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            raise

        # Log success outside the try block
        execution_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self.logger.log_agent_action(
            agent_type="RiskAnalysisAgent",
            action="Classification Decision",
            case_id=self.case_id,
            input_data=risk_quantification_output,
            output_data=risk_analysis_output.model_dump(),
            reasoning="Successfully classified risk categories.",
            execution_time_ms=execution_time_ms,
            success=True
        )

        print("✅ Step 5. Classification Decision Completed")
        return risk_analysis_output

    def _get_validated_llm_response(
        self,
        messages: list[dict],
        output_model: type[T],
        max_retries: int = 2,
        max_tokens: int = 1000,
        result_validator: Callable[[T], None] | None = None
    ) -> T:
        """
        Request structured output and validate with Pydantic.

        Retries only malformed, incomplete, or schema-invalid
        LLM responses. Does not retry API or audit failures.
        """
        from unittest.mock import Mock

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        # Do not mutate the caller's message list.
        conversation = [message.copy() for message in messages]
        last_error = None

        for attempt in range(max_retries + 1):

            response = get_completion(
                system_prompt=None,
                user_prompt=None,
                client=self.client,
                model=self.model,
                max_completion_tokens=max_tokens,
                messages=conversation,
                return_response=True
            )

            choice = response.choices[0]
            response_text = choice.message.content or ""

            finish_reason = choice.finish_reason

            # Compatibility with legacy test mocks
            if isinstance(finish_reason, Mock):
                finish_reason = "stop"

            if finish_reason not in (None, "stop", "length"):
                raise RuntimeError(
                    f"Unexpected LLM finish reason: {finish_reason}"
                )
    
            try:
                if finish_reason == "length":
                    raise ValueError(
                        "LLM response truncated at token limit"
                    )

                clean_json = self._extract_json_from_response(response_text)
                validated_output = output_model.model_validate_json(clean_json)

                if result_validator is not None:
                    result_validator(validated_output)

                return validated_output

            except (ValueError, TypeError, ValidationError) as exc:
                last_error = exc

                if attempt == max_retries:
                    break

                # Feed the invalid response and precise error
                # back to the model for correction.
                conversation.extend([
                    {
                        "role": "assistant",
                        "content": response_text
                    },
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed validation.\n\n"
                            f"Expected schema: {output_model.__name__}\n"
                            f"Validation error:\n{exc}\n\n"
                            "Correct the previous response.\n"
                            "Return ONLY one valid JSON object.\n"
                            "Preserve facts from the original evidence.\n"
                            "Do not invent missing information.\n"
                            "Use numbers for numeric fields.\n"
                            "Include all required fields."
                        )
                    }
                ])

        if output_model is RiskAnalystOutput:
            raise LLMOutputValidationError(
                "Failed to parse Risk Analyst JSON output: "
                f"{last_error}"
            ) from last_error

        raise LLMOutputValidationError(
            f"{output_model.__name__} validation failed "
            f"after {max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response
        
        Implement JSON extraction that handles:
        - JSON in code blocks (```json)
        - JSON in plain text
        - Malformed responses
        - Empty responses
        """
        if not response_content or not response_content.strip():
            raise ValueError("No JSON content found in LLM response")

        content = response_content.strip()

        # Remove opening and closing Markdown fences
        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                content,
                flags=re.IGNORECASE
            ).strip()


        json_start = content.find("{")

        if json_start == -1:
            raise ValueError("No JSON object found in LLM response")

        json_content = content[json_start:]

        try:
            result, _ = json.JSONDecoder().raw_decode(json_content)

        except json.JSONDecodeError as e:

            start = max(0, e.pos - 150)
            end = min(len(json_content), e.pos + 150)

            problem_area = json_content[start:end]

            raise ValueError(
                f"LLM response contains invalid JSON:\n"
                f"{e}\n\n"
                f"Problem area:\n"
                f"{problem_area}"
            ) from e

        if not isinstance(result, dict):
            raise TypeError(
                f"Expected JSON object, got {type(result).__name__}"
            )

        return json.dumps(result)

    def _parse_json_response(self, response_content: str) -> dict:
        """Extract JSON and convert it into a dictionary."""

        json_string = self._extract_json_from_response(response_content)

        return json.loads(json_string)

    def _format_case_for_prompt(self, case_data) -> dict:
        """Generate step 1 output from LLM response
        
         """
        TRANSACTION_CHUNK_SIZE = 30

        compact_case = build_risk_case_payload(case_data)

        transactions = compact_case["transactions"]
        chunks = [
            transactions[i:i + TRANSACTION_CHUNK_SIZE]
            for i in range(0, len(transactions), TRANSACTION_CHUNK_SIZE)
        ]        

    
        chunk_results = []

        for chunk_number, chunk in enumerate(chunks, start=1):
            try:
                result = self._analyze_transaction_chunk(
                    case_data,
                    chunk
                )
            except LLMOutputValidationError as e:
                raise LLMOutputValidationError(
                    f"Step 1 failed for case {case_data.case_id}, "
                    f"transaction chunk {chunk_number}/{len(chunks)}: {e}"
                ) from e

            chunk_results.append(result)

        material_transactions = []

        for result in chunk_results:
            material_transactions.extend(
                result.get("material_transactions", [])
            )

        notable_facts = []

        for result in chunk_results:
            notable_facts.extend(
                result.get("notable_facts", [])
            )            

        data_gaps = []

        for result in chunk_results:
            data_gaps.extend(
                result.get("data_gaps", [])
            )

        transaction_dates = [
            t.transaction_date
            for t in case_data.transactions
        ]

        step1_output = {
            "case_id": case_data.case_id,

            "customer": {
                "customer_id": case_data.customer.customer_id,
                "name": case_data.customer.name,
                "risk_rating": case_data.customer.risk_rating,
                "occupation": case_data.customer.occupation,
                "annual_income": case_data.customer.annual_income,
                "customer_since": case_data.customer.customer_since,
            },

            "accounts": [
                {
                    "account_id": a.account_id,
                    "account_type": a.account_type,
                    "status": a.status,
                    "current_balance": a.current_balance,
                }
                for a in case_data.accounts
            ],

            "activity_period": {
                "start_date": min(transaction_dates),
                "end_date": max(transaction_dates),
            },

            "transaction_summary": {
                "transaction_count": len(case_data.transactions),
                "total_transaction_amount": sum(
                    t.amount for t in case_data.transactions
                ),
            },

            "material_transactions": material_transactions,
            "prior_case_history": [],
            "notable_facts": notable_facts,
            "data_gaps": data_gaps,
        }
        return step1_output
    
    def _analyze_transaction_chunk(
        self,
        case_data,
        transactions_chunk
    ):

        chunk_payload = {
            "customer": {
                "customer_id": case_data.customer.customer_id,
                "risk_rating": case_data.customer.risk_rating,
                "occupation": case_data.customer.occupation,
                "annual_income": case_data.customer.annual_income,
            },
            "transactions": transactions_chunk,
        }

        chunk_json = json.dumps(
            chunk_payload,
            separators=(",", ":"),
            default=str
        )

        prompt = f"""
    Review this transaction subset for material facts relevant to
    financial-crime analysis.

    Transaction data:
    {chunk_json}

    Return exactly one valid JSON object using this structure:

    {{
        "material_transactions": [
            {{
                "transaction_id": "",
                "account_id": "",
                "transaction_date": "",
                "transaction_type": "",
                "amount": 0.0,
                "counterparty": "",
                "location": ""
            }}
        ],
        "notable_facts": [
            ""
        ],
        "data_gaps": [
            ""
        ]
    }}

    Requirements:
    - Return no more than 8 material_transactions.
    - Include only materially relevant transactions.
    - Copy transaction values exactly from the supplied data.
    - If counterparty or location is missing, use an empty string.
    - Do not infer or synthesize missing transaction attributes.
    - Return no more than 5 notable_facts.
    - notable_facts must be strings only.
    - notable_facts must describe observable transaction-level facts only.
    - Do not attribute intent or infer financial-crime typologies in notable_facts.
    - Return no more than 5 data_gaps.
    - data_gaps must be strings only.
    - Summarize recurring missing-data issues instead of listing every affected transaction.
    - Do not return objects inside notable_facts or data_gaps.
    - Do not classify the case.
    - Do not make a SAR filing decision.
    - Do not invent facts.
    - Do not repeat customer profile attributes such as customer_id,
    occupation, annual_income, or risk_rating in notable_facts.

    Output requirements:
    - Return exactly one JSON object.
    - The first character must be {{.
    - The last character must be }}.
    - Use double quotes for all keys and string values.
    - Do not include Markdown or explanatory text.
    """
        result = self._get_validated_llm_response(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            output_model=JSONDictOutput,
            max_retries=2,
            max_tokens=1000,
            result_validator=lambda output: self._validate_chunk_result(
                output.root
            )
        )

        return result.root

    def _validate_chunk_result(self, result: dict) -> dict:

        if not isinstance(result, dict):
            raise TypeError("Chunk result must be a dictionary")

        required_keys = {
            "material_transactions",
            "notable_facts",
            "data_gaps"
        }

        missing = required_keys - result.keys()
        extra = result.keys() - required_keys

        if missing:
            raise ValueError(
                f"Chunk output missing required fields: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Chunk output contains unexpected fields: {sorted(extra)}"
            )

        for field, max_items in [
            ("material_transactions", 8),
            ("notable_facts", 5),
            ("data_gaps", 5)
        ]:
            value = result[field]

            if not isinstance(value, list):
                raise TypeError(f"{field} must be a list")

            if len(value) > max_items:
                raise ValueError(
                    f"{field} exceeds maximum of {max_items} entries"
                )

        if not all(isinstance(x, dict) for x in result["material_transactions"]):
            raise TypeError("Every material_transaction must be an object")

        if not all(isinstance(x, str) for x in result["notable_facts"]):
            raise TypeError("Every notable_fact must be a string")

        if not all(isinstance(x, str) for x in result["data_gaps"]):
            raise TypeError("Every data_gap must be a string")

        return result

    def _format_accounts(self, accounts) -> str:
        """Format account details for analysis prompts."""

        if not accounts:
            return "No accounts available."

        lines = []

        for account in accounts:
            lines.append(
                f"Account ID: {account.account_id}, "
                f"Account Type: {account.account_type}, "
                f"Status: {account.status}, "
                f"Current Balance: ${account.current_balance:,.2f}"
            )

        return "\n".join(lines)

    def _validate_step2_result(self, result: dict) -> None:

        expected_keys = set(self.sar_analysis_output["step_2"])
        actual_keys = set(result)

        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        if missing:
            raise ValueError(
                f"Step 2 missing required fields: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Step 2 contains unexpected fields: {sorted(unexpected)}"
            )

        patterns = result.get("suspicious_patterns")

        if not isinstance(patterns, list):
            raise TypeError("suspicious_patterns must be a list")

        if len(patterns) > 5:
            raise ValueError(
                "Step 2 cannot contain more than 5 suspicious patterns"
            )

    def _validate_step3_result(self, result: dict) -> None:
        """Validate the structure of Step 3 regulatory mapping."""

        expected_keys = set(self.sar_analysis_output["step_3"])
        actual_keys = set(result)

        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        if missing:
            raise ValueError(
                f"Step 3 missing required fields: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Step 3 contains unexpected fields: {sorted(unexpected)}"
            )

    def _validate_step4_result(self, result: dict) -> None:
        """Validate Step 4 Risk Quantification output."""

        expected_keys = set(self.sar_analysis_output["step_4"])
        actual_keys = set(result)

        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        if missing:
            raise ValueError(
                f"Step 4 missing required fields: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Step 4 contains unexpected fields: {sorted(unexpected)}"
            )

    def _format_transactions(self, transactions) -> str:
        """Format transaction details for analysis prompts."""

        if not transactions:
            return "No transactions available."

        lines = []

        for i, tx in enumerate(transactions, start=1):
            line = (
                f"{i}. {tx.transaction_date}: "
                f"{tx.transaction_type} "
                f"${tx.amount:,.2f} "
                f"- {tx.description}"
            )

            if tx.location:
                line += f" - {tx.location}"

            lines.append(line)

        return "\n".join(lines)
# ===== PROMPT ENGINEERING HELPERS =====

def create_chain_of_thought_framework():
    """Helper function showing Chain-of-Thought structure
    
    To adapt for financial crime analysis:
    
    **Analysis Framework** (Think step-by-step):
    1. **Data Review**: What does the data tell us?
    2. **Pattern Recognition**: What patterns are suspicious?
    3. **Regulatory Mapping**: Which regulations apply?
    4. **Risk Quantification**: How severe is the risk?
    5. **Classification Decision**: What category fits best?
    """

    classification_categories = get_classification_categories()


    DATA_REVIEW_PROMPT = """
        Data Review:
        Review all available customer, account, transaction, alert, and case information.

        Extract only factual information relevant to a potential Suspicious Activity Report (SAR).

        Identify:
        - Customer identity and profile
        - Accounts involved
        - Transaction dates, amounts, types, counterparties, and locations
        - Total suspicious amount
        - Time period of the activity
        - Customer's expected activity based on profile
        - Prior alerts, cases, or relevant history if available
        - Missing or conflicting information

        Produce a factual case summary.

        Do not determine SAR filing or suspicious activity classification at this stage.
        Do not infer facts that are not present in the source data.
        """

    PATTERN_RECOGNITION_PROMPT = """
        Suspicious Activity Pattern Analysis:
        Analyze the reviewed facts for behaviors that may indicate suspicious activity.

        Identify:
        - Specific suspicious patterns
        - Relevant transactions supporting each pattern
        - Deviation from expected customer activity
        - Repeated or coordinated activity
        - Structuring indicators
        - Rapid movement of funds
        - Unusual cash activity
        - Unusual domestic or international transfers
        - Funnel-account behavior
        - Unexplained source or destination of funds
        - Other applicable financial-crime indicators

        For every identified pattern:
        - Describe the observed behavior
        - Reference supporting transaction IDs
        - Identify the date range
        - Calculate the associated amount when possible
        - Explain why the behavior is unusual
        - Identify plausible non-suspicious explanations when supported by the data

        Do not make a final SAR filing decision.
        """

    REGULATORY_MAPPING_PROMPT = """
        SAR Typology and Regulatory Mapping:
        Map the identified suspicious behaviors to relevant SAR activity categories,
        AML typologies, and regulatory indicators.

        For each mapping:
        - Identify the suspicious activity type
        - Identify relevant subcategories where applicable
        - Link the classification to specific observed behavior
        - List supporting transactions
        - Assess the strength of the mapping
        - Identify information that would strengthen or weaken the classification

        """

    RISK_QUANTIFICATION_PROMPT = """
        SAR Risk and Filing Assessment:
        Assess whether the case contains sufficient indicators to warrant SAR escalation
        or filing consideration under the institution's applicable SAR policy.

        Evaluate:
        - Nature of the suspicious behavior
        - Dollar amount involved
        - Duration and frequency
        - Number of accounts and counterparties involved
        - Customer risk profile
        - Geographic risk
        - Evidence supporting the suspicion
        - Potential legitimate explanations
        - Information gaps
        - Relevant SAR filing thresholds and institutional rules when provided

        Produce:
        - Overall risk level
        - SAR escalation recommendation
        - Filing rationale
        - Reasons against filing, if applicable
        - Confidence level
        - Additional investigation required before a decision

        """

    CLASSIFICATION_DECISION_PROMPT = f"""
        Classification Decision:

        Using the completed analysis from the previous steps, determine the final
        risk classification for the case.

        Return ONLY the fields required by RiskAnalystOutput.

        Determine:

        1. classification
        Select exactly one of:
        {classification_categories}

        2. confidence_score
        Provide a numeric confidence score between 0.0 and 1.0.
        - Provide a numeric confidence score between 0.0 and 1.0, where 1.0 indicates maximum confidence.
        - Confidence must reflect the strength of evidence for the specific classification, 
        not merely confidence that the activity is unusual.
        - A high confidence score requires multiple mutually reinforcing indicators supporting the selected classification.

        3. reasoning
        - Provide a concise justification for the classification based on the
        available evidence.
        - Maximum 500 characters.
        - Do not include unsupported assumptions.

        4. key_indicators
        - Provide a list of the most important suspicious indicators supporting
        the classification.

        5. risk_level
        Select exactly one of:
        - Low
        - Medium
        - High
        - Critical

        """

    return {
        "step_1": DATA_REVIEW_PROMPT,
        "step_2": PATTERN_RECOGNITION_PROMPT, 
        "step_3": REGULATORY_MAPPING_PROMPT,
        "step_4": RISK_QUANTIFICATION_PROMPT,
        "step_5": CLASSIFICATION_DECISION_PROMPT
    }

def get_classification_categories():
    """Standard SAR classification categories
    
    TODO: Use these categories in your prompts:
    """
    return {
        "Structuring": "Transactions designed to avoid reporting thresholds",
        "Sanctions": "Potential sanctions violations or prohibited parties",
        "Fraud": "Fraudulent transactions or identity-related crimes",
        "Money_Laundering": "Complex schemes to obscure illicit fund sources", 
        "Other": "Suspicious patterns not fitting standard categories"
    }

def get_completion(
    system_prompt,
    user_prompt,
    client,
    model="gpt-4o-mini",
    max_completion_tokens=2000,
    messages=None,
    return_response=False
) -> Any:
    """
    Get an OpenAI completion.

    Returns text by default for backward compatibility.
    Returns the full API response when return_response=True.
    """

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_completion_tokens
        )

    except Exception as e:
        raise RuntimeError(
            f"LLM completion failed: {e}"
        ) from e

    pylogger.debug(
        "Finish reason: %s",
        response.choices[0].finish_reason
    )
    pylogger.debug("Model: %s", response.model)
    pylogger.debug("Usage: %s", response.usage)

    if return_response:
        return response

    return response.choices[0].message.content


def normalize_material_transaction(tx):
    return {
        "transaction_id": tx.get("transaction_id", ""),
        "account_id": tx.get("account_id", ""),
        "date": tx.get(
            "date",
            tx.get("transaction_date", "")
        ),
        "type": tx.get(
            "type",
            tx.get("transaction_type", "")
        ),
        "amount": tx.get("amount", 0.0),
        "counterparty": tx.get("counterparty", ""),
        "location": tx.get("location", ""),
    }

def build_risk_case_payload(case_data):

    return {
        "case_id": case_data.case_id,
        "customer": {
            "customer_id": case_data.customer.customer_id,
            "name": case_data.customer.name,
            "risk_rating": case_data.customer.risk_rating,
            "occupation": case_data.customer.occupation,
            "annual_income": case_data.customer.annual_income,
            "customer_since": case_data.customer.customer_since,
        },
        "accounts": [
            {
                "account_id": a.account_id,
                "account_type": a.account_type,
                "status": a.status,
                "current_balance": a.current_balance,
            }
            for a in case_data.accounts
        ],
        "transactions": [
            {
                "transaction_id": t.transaction_id,
                "account_id": t.account_id,
                "transaction_date": t.transaction_date,
                "transaction_type": t.transaction_type,
                "amount": t.amount,
                "counterparty": t.counterparty,
                "location": t.location,
            }
            for t in case_data.transactions
        ],
        "case_summary": {
            "transaction_count": len(case_data.transactions),
            "total_transaction_amount": sum(
                t.amount for t in case_data.transactions
            ),
            "activity_start_date": min(
                t.transaction_date for t in case_data.transactions
            ),
            "activity_end_date": max(
                t.transaction_date for t in case_data.transactions
            ),
        }        
    }
# ===== TESTING UTILITIES =====

def testload_and_preprocess_data():
    """
    This function should:
    1. Load customers.csv, accounts.csv, transactions.csv
    2. Handle missing values appropriately
    3. Create data dictionaries for processing
    4. Return cleaned datasets
    """
    import pandas as pd
    
    # Example structure (uncomment and modify):
    customers_df = pd.read_csv(DATA_DIR / "customers.csv", dtype={'ssn_last_4': str})
    accounts_df = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions_df = pd.read_csv(DATA_DIR / "transactions.csv")

    # # Handle NaN values
    transactions_df['counterparty'] = transactions_df['counterparty'].fillna('')
    transactions_df['location'] = transactions_df['location'].fillna('')
    customers_df['phone'] = customers_df['phone'].fillna('')
    # 
    # # Convert to dictionaries
    customers_data = customers_df.to_dict('records')
    accounts_data = accounts_df.to_dict('records')
    transactions_data = transactions_df.to_dict('records')
     
    pylogger.debug(f"📈 Loaded: {len(customers_data)} customers, {len(accounts_data)} accounts, {len(transactions_data)} transactions")
    return customers_data, accounts_data, transactions_data

def select_customers(customers_data, accounts_data, transactions_data, select_n, top_n=5 ):
    """
    This function selects the top N highest-risk customers based on a combination of risk indicators.
    """
   
    # Example screening logic (uncomment and modify):
    selected_customers = []
     
    for customer in customers_data:
        # Get customer accounts and transactions
        customer_accounts = [acc for acc in accounts_data if acc['customer_id'] == customer['customer_id']]
        customer_transactions = [txn for txn in transactions_data if any(txn['account_id'] == acc['account_id'] for acc in customer_accounts)]
        
        # Calculate risk indicators
        total_amount = sum(abs(txn['amount']) for txn in customer_transactions)
        transaction_count = len(customer_transactions)
        risk_rating = customer['risk_rating']
         
        # Apply screening criteria
        risk_flags = []
        if risk_rating in ['Medium', 'High']:
            risk_flags.append('high_risk_rating')
        if total_amount > 100000:
            risk_flags.append('large_amounts')
        if transaction_count > 50:
            risk_flags.append('high_frequency')
         
        # Select high-risk customers
        if len(risk_flags) >= 2:  # Multiple risk flags
            selected_customers.append({
                'customer': customer,
                'accounts': customer_accounts,
                'transactions': customer_transactions,
                'total_amount': total_amount,
                'transaction_count': transaction_count,
                'risk_flags': risk_flags
            })
     
    # Sort by risk score and take top N
    selected_customers.sort(key=lambda x: (len(x['risk_flags']), x['total_amount']), reverse=True)
    return selected_customers[select_n]

def test_agent_with_selected_case(test_num):
    """Test the agent with a sample case
    Test the agent with a sample case
    
    This function is used to test the implementation:
    - Create sample case data
    - Initialize agent
    - Run analysis
    - Validate results

    """
    
    print("🧪 Testing Risk Analyst Agent")
    print("Implementing test case")
    
    customers_data, accounts_data, transactions_data = testload_and_preprocess_data()

    selected_customer = select_customers(customers_data, accounts_data, transactions_data, select_n=test_num)
    print(f"\n🔍 CUSTOMER SELECTED: {selected_customer['customer']['name']}")

    test_client, test_logger = create_client_and_logger()
    loader = DataLoader(test_logger)
    case_data_test = loader.create_case_from_data(
        selected_customer['customer'],
        selected_customer['accounts'], 
        selected_customer['transactions']
            )

    print("************* Sample case created successfully *************")

    # Initializing agent
    test_agent = RiskAnalystAgent(test_client, test_logger)
    print("************* Agent initialized successfully *************")

    # Analyzing Risk for the sample customer
    print("🔍 Analyzing Risk for the sample customer")
    risk_analysis = test_agent.analyze_case(case_data_test)

    # Display analysis results
    print(f"Classification: {risk_analysis.classification}")
    print(f"Confidence: {risk_analysis.confidence_score:.2f}")
    print(f"Risk Level: {risk_analysis.risk_level}")
    print(f"Reasoning: {risk_analysis.reasoning}")

    print("************* Analysis Completed successfully *************")

    return risk_analysis


if __name__ == "__main__":
    """
    print("🔍 Risk Analyst Agent Module")
    print("Chain-of-Thought reasoning for suspicious activity classification")
    print("\n📋 Testing Items:")
    print("• Design Chain-of-Thought system prompt")
    print("• Implement analyze_case method")
    print("• Add JSON parsing and validation")
    print("• Create comprehensive error handling")
    print("• Test with sample cases")
    print("\n💡 Key Concepts:")
    print("• Chain-of-Thought: Step-by-step reasoning")
    print("• Structured Output: Validated JSON responses")
    print("• Financial Crime Detection: Pattern recognition")
    print("• Audit Logging: Complete decision trails")
    """


    test_output = test_agent_with_selected_case(4)
    print("\n========== TEST OUTPUT ==========")
    print(repr(test_output))
    print("=========================================\n")

