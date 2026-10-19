# Compliance Officer Agent - ReACT Implementation  
# Implement Compliance Officer Agent using ReACT prompting

"""
Compliance Officer Agent Module

This agent generates regulatory-compliant SAR narratives using ReACT prompting.
It takes risk analysis results and creates structured documentation for 
FinCEN submission.

YOUR TASKS:
- Study ReACT (Reasoning + Action) prompting methodology
- Design system prompt with Reasoning/Action framework
- Implement narrative generation with word limits
- Validate regulatory compliance requirements
- Create proper audit logging and error handling
"""

import json
import re
import os
from openai import OpenAI
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid
from pprint import pprint
from pydantic import ValidationError
from pathlib import Path

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
        print("Get your Vocareum OpenAI API key from 'Cloud Resources' in your workspace")
    else:
        client = OpenAI(
            api_key=openai_api_key
        )
        print(f"🔑 API key: {openai_api_key[:8]}...{openai_api_key[-4:]}")

    return client, explainability_logger


# Importing foundation components

from .foundation_sar import (
    CustomerData,
    AccountData,
    TransactionData,
    CaseData,
    RiskAnalystOutput,
    ComplianceOfficerOutput,
    ExplainabilityLogger,
    DataLoader,
    load_csv_data
)

import logging
from dotenv import load_dotenv

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

class ComplianceOfficerAgent:
    """
    Compliance Officer agent using ReACT prompting framework.
    
    Implement agent that:
    - Uses Reasoning + Action structured prompting
    - Generates regulatory-compliant SAR narratives
    - Enforces word limits and terminology
    - Includes regulatory citations
    - Validates narrative completeness
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-4.1-mini"):
        """Initialize the Compliance Officer Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        # Initialize agent components
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        
        # Design ReACT system prompt
        React_framework = create_react_framework()

        self.reasoning_phase = React_framework["reasoning_phase"]
        self.action_phase = React_framework["action_phase"]

        self.system_prompt = """
        You are a senior Compliance Officer specializing in BSA/AML
        and FinCEN Suspicious Activity Reports (SAR).

        TASK:
        Generate a factual, regulatory-compliant SAR narrative using ReACT
        (Reasoning and Acting).

        Use the following reasoning guidance internally:
        {self.reasoning_phase}

        Use the following narrative-generation guidance:
        {self.action_phase}

        NARRATIVE REQUIREMENTS:
        - Maximum 120 words.
        - Begin with "Customer <name>".
        - Include supported dates, amounts, accounts, and relevant parties.
        - Describe the suspicious activity and explain why it is unusual.
        - Use appropriate regulatory terminology, such as
        "Bank Secrecy Act (BSA)" or "suspicious activity".
        - Do not invent transactions, infer intent, or speculate.
        - Base all statements exclusively on the supplied case evidence.

        AVAILABLE TOOLS:
        1. get_regulatory_requirements
        Parameters: {}

        2. validate_word_count
        Parameters: {"text": "<narrative>", "max_words": 120}

        3. _validate_narrative_compliance
        Parameters: {"narrative": "<draft narrative>"}

        ReACT WORKFLOW:

        1. Draft a narrative internally.

        2. Request validation by returning ONLY this JSON:
        {
            "tool": "_validate_narrative_compliance",
            "parameters": {"narrative": "<draft narrative>"}
        }

        3. Wait for the OBSERVATION containing the validation result.

        4. If "valid" is false:
        - Identify the failed validation checks.
        - Correct only the relevant issues.
        - Request validation again.

        5. If "valid" is true:
        - Immediately return FINAL.
        - Do not call another tool.
        - Preserve the validated narrative EXACTLY.

        FINAL OUTPUT:
        Return ONLY a JSON object conforming to ComplianceOfficerOutput,
        with these fields:
        - narrative
        - narrative_reasoning
        - regulatory_citations
        - completeness_check

        Do not include Markdown fences, THOUGHT text, ACTION labels,
        or explanations outside the required JSON.

        Never return FINAL before successful narrative validation.
        """

    def generate_compliance_narrative(self, case_data, risk_analysis) -> ComplianceOfficerOutput:

        risk_analysis_str = self._format_risk_analysis_for_prompt(risk_analysis)
        case_data_json = case_data.model_dump_json(indent=2)

        messages = [
            {
            "role": "system",
            "content": self.system_prompt
        },
            {
            "role": "user",
            "content": f"""
        Case Data:
        {case_data_json}

        Risk Analysis Output:
        {risk_analysis_str}

        Generate a regulatory-compliant compliance narrative using the ReACT process.

        IMPORTANT:
        Every narrative MUST be validated before FINAL is returned.

        FIRST STEP:
        - Draft the narrative internally.
        - Do not output the draft separately.
        - Call _validate_narrative_compliance.
        - Put the complete draft inside the narrative parameter.

        Your first response MUST be:

        ACTION:

        ```json
        {{
            "tool": "_validate_narrative_compliance",
            "parameters": {{
                "narrative": "<draft narrative>"
            }}
        }}

        After receiving an OBSERVATION:

        If "valid" is false, revise the narrative and call
        _validate_narrative_compliance again.
        If "valid" is true, immediately return FINAL.

        Do not return FINAL until validation succeeds.

        FINAL must conform exactly to:

        {json.dumps(ComplianceOfficerOutput.model_json_schema(), indent=2)}
        """
        }]

        max_iterations = 10  # Define the maximum number of ReACT iterations
        validated_narrative = None
    
        for iteration in range(max_iterations):

            response_text = get_completion(
                messages=messages,
                client=self.client,
                model=self.model
            )

            pylogger.info(f"\n--- ReACT Iteration {iteration + 1} ---")
            pylogger.debug(response_text)

            # -----------------------------------
            # 1. Check for FINAL output
            # -----------------------------------

            final_output = self._extract_final_json(response_text)

            if final_output is not None:
                output = ComplianceOfficerOutput.model_validate(final_output)

                if validated_narrative is None:

                    # Independently enforce the word limit.
                    word_count = len(output.narrative.split())

                    if word_count > 120:
                        raise ValueError(
                            f"Narrative exceeds 120 word limit: {word_count} words"
                        )

                    # Accept direct FINAL after basic validation.
                    validated_narrative = output.narrative

                if output.narrative != validated_narrative:
                    raise ValueError(
                        "FINAL narrative differs from the validated narrative"
                    )

                self.logger.log_agent_action(
                    agent_type="ComplianceOfficer",
                    action="Generate Compliance Narrative",
                    case_id=case_data.case_id,
                    input_data=case_data.model_dump(),
                    output_data=output.model_dump(),
                    reasoning="Narrative generated and validated successfully",
                    execution_time_ms=0,
                    success=True
)
                return output
            # -----------------------------------
            # 2. Extract ACTION
            # -----------------------------------
            action = self._extract_action(response_text)

            if action is None:
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })

                messages.append({
                    "role": "user",
                    "content": """
                    PROTOCOL ERROR:

                    Return either:
                    1. An ACTION containing a valid JSON tool request.
                    2. A FINAL ComplianceOfficerOutput JSON object,
                    but only after successful validation.
                    """
                })

                continue  # Important: start the next ReACT iteration

            # -----------------------------------
            # 3. Execute tool
            # -----------------------------------

            tool_result = self._execute_tool(
                action["tool"],
                action.get("parameters", {})
            )

            print(
                "\nTOOL RESULT:",
                json.dumps(tool_result, indent=2, default=str)
            )

            if action["tool"] == "_validate_narrative_compliance":
                if tool_result.get("valid") is True:
                    validated_narrative = action["parameters"]["narrative"]
                else:
                    validated_narrative = None
            # -----------------------------------
            # 4. Give observation back to LLM
            # -----------------------------------

            messages.append({
                "role": "assistant",
                "content": response_text
            })

            messages.append({
                "role": "user",
                "content": f"""
                OBSERVATION:
                {json.dumps(tool_result, indent=2)}

                IMPORTANT:

                If this was a narrative validation result:

                - If valid is false, identify which validation checks failed.
                Correct those specific problems and request validation again.

                - If valid is true, immediately return FINAL as a JSON object
                conforming to ComplianceOfficerOutput.

                - The FINAL narrative MUST be exactly the same as the
                successfully validated narrative.

                Do not repeatedly paraphrase a narrative that has already
                passed validation.

                If this was a non-validation tool result, continue the
                ReACT process using the information returned.
                """
            })

        # ReACT exhausted without producing a valid FINAL response.

        error_message = "Failed to parse Compliance Officer JSON output"

        self.logger.log_agent_action(
            agent_type="ComplianceOfficer",
            action="Generate Compliance Narrative",
            case_id=case_data.case_id,
            input_data=case_data.model_dump(),
            output_data={},
            reasoning="JSON parsing failed: " + error_message,
            execution_time_ms=0,
            success=False
        )

        raise ValueError(error_message)

    def _execute_tool(self, tool_name: str, parameters: dict) -> dict:

        if tool_name == "get_regulatory_requirements":
            return get_regulatory_requirements()
        elif tool_name == "validate_word_count":
            text = parameters.get("text", "")
            max_words = parameters.get("max_words", 120)

            valid = validate_word_count(
                text,
                max_words=max_words
            )

            return {
                "valid": valid,
                "word_count": len(text.split()),
                "max_words": max_words
            }

        elif tool_name == "_validate_narrative_compliance":
            narrative = parameters.get("narrative", "")

            return self._validate_narrative_compliance(
                narrative
            )
        else:
            raise ValueError(
                f"Unknown tool requested: {tool_name}"
            )
    
    def _extract_action(self, response_content: str) -> dict | None:

        if not isinstance(response_content, str):
            return None

        content = response_content.strip()

        if not content:
            return None

        json_blocks = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            content,
            re.DOTALL | re.IGNORECASE
        )

        candidates = json_blocks + [content]

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if (
                isinstance(data, dict)
                and isinstance(data.get("tool"), str)
                and isinstance(data.get("parameters"), dict)
            ):
                return data

        return None

    def _extract_final_json(self, response_content: str) -> dict | None:
        """
        Extract a final ComplianceOfficerOutput from either:
        1. Plain JSON
        2. JSON in Markdown code fences
        3. FINAL: followed by JSON
        """

        if not isinstance(response_content, str) or not response_content.strip():
            return None

        content = response_content.strip()

        # Handle optional FINAL marker
        if content.startswith("FINAL:"):
            content = content[len("FINAL:"):].strip()

        candidates = []

        # Extract fenced JSON, if present
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            content,
            re.DOTALL | re.IGNORECASE
        )

        candidates.extend(fenced_blocks)

        # Also attempt to parse plain JSON
        candidates.append(content)

        for candidate in candidates:
            try:
                data = json.loads(candidate)

                if not isinstance(data, dict):
                    continue

                # Do not mistake a tool request for final output
                if "tool" in data:
                    continue

                # Validate against the Pydantic model
                validated = ComplianceOfficerOutput.model_validate(data)

                return validated.model_dump()

            except (json.JSONDecodeError, ValidationError):
                continue

        return None

    def _extract_json_from_response(self, response_text: str) -> str:

        if not response_text or not response_text.strip():
            raise ValueError("No JSON content found")

        result = self._extract_final_json(response_text)

        if result is None:
            raise ValueError("No JSON content found")

        return json.dumps(result)

    def _format_transactions_for_compliance(self, transactions) -> str:

        formatted = []

        for i, txn in enumerate(transactions, start=1):

            line = (
                f"{i}. {txn.transaction_date}: "
                f"${txn.amount:,.2f} {txn.transaction_type}"
            )

            if txn.location:
                line += f" at {txn.location}"

            if txn.method:
                line += f" via {txn.method}"

            formatted.append(line)

        return "\n".join(formatted)
   
    def _format_risk_analysis_for_prompt(self, risk_analysis) -> str:
        """Format risk analysis results for compliance prompt
        
        Create structured format that includes:
        - Classification and confidence
        - Key suspicious indicators
        - Risk level assessment
        - Analyst reasoning
        """
        indicators = "\n".join(
            f"- {indicator}"
            for indicator in risk_analysis.key_indicators
        )

        return f"""
            RISK ANALYSIS RESULTS

            Classification: {risk_analysis.classification}
            Confidence Score: {risk_analysis.confidence_score:.2f}
            Risk Level: {risk_analysis.risk_level}

            Key Suspicious Indicators:
            {indicators}

            Analyst Reasoning:
            {risk_analysis.reasoning}
            """.strip()        

    def _validate_narrative_compliance(self, narrative: str) -> Dict[str, Any]:
        """Validate narrative meets regulatory requirements
        
        Implement validation that checks:
        - Word count (≤120 words)
        - Required elements present
        - Appropriate terminology
        - Regulatory completeness
        """
        # Get regulatory requirements
        requirements = get_regulatory_requirements()

        # 1. Validate word count
        word_limit = requirements["word_limit"]
        word_count = len(narrative.split())

        word_count_valid = validate_word_count(
            narrative,
            max_words=word_limit
        )

        narrative_lower = narrative.lower()

        # 2. Check required narrative elements
        element_checks = {
            "Customer identification": any(
                term in narrative_lower
                for term in ["customer", "client", "account holder"]
            ),

            "Suspicious activity description": any(
                term in narrative_lower
                for term in ["suspicious", "unusual", "structuring", "fraud"]
            ),

            "Transaction amounts and dates": (
                "$" in narrative
                or "transaction" in narrative_lower
                or "deposit" in narrative_lower
                or "transfer" in narrative_lower
            ),

            "Why activity is suspicious": any(
                term in narrative_lower
                for term in [
                    "suspicious because",
                    "considered suspicious",
                    "appears suspicious",
                    "pattern",
                    "inconsistent",
                    "deviates from",
                    "deviation from",
                    "unusual for",
                    "lack of legitimate",
                    "structuring"
                ]
                
            )
        }

        required_elements_valid = all(element_checks.values())

        # 3. Check regulatory terminology

        terminology_equivalents = {
            "Suspicious activity": [
                "suspicious activity", "structuring"
            ],
            "Regulatory threshold": [
                "regulatory threshold",
                "reporting threshold",
                "ctr threshold"
            ],
            "Financial institution": [
                "financial institution", "bank"
            ],
            "Money laundering": [
                "money laundering"
            ],
            "Bank Secrecy Act": [
                "bank secrecy act", "bsa"
            ]
        }

        terminology_found = [
            term
            for term, alternatives in terminology_equivalents.items()
            if any(alt in narrative_lower for alt in alternatives)
        ]

        terminology_valid = len(terminology_found) > 0



        # 4. Regulatory completeness
        regulatory_complete = (
            word_count_valid
            and required_elements_valid
            and terminology_valid
        )

        return {
            "valid": regulatory_complete,
            "word_count": word_count,
            "word_limit": word_limit,
            "word_count_valid": word_count_valid,
            "required_elements": element_checks,
            "required_elements_valid": required_elements_valid,
            "terminology_found": terminology_found,
            "terminology_valid": terminology_valid,
            "regulatory_complete": regulatory_complete,
            "regulatory_citations": requirements["citations"]
        }

# ===== REACT PROMPTING HELPERS =====

def create_react_framework():
    """Helper function showing ReACT structure for compliance narrative generation.

    **REASONING Phase:**
    1. Review the risk analyst's classification, reasoning, and key indicators
    2. Identify the material facts, transactions, dates, amounts, and parties
    3. Determine which facts support the suspicious activity classification
    4. Organize the facts into a clear chronological compliance narrative
    5. Ensure the narrative is factual, concise, and does not introduce unsupported information

    **ACTION Phase:**
    1. Draft a concise compliance narrative (≤120 words)
    2. State the relevant activity, dates, amounts, accounts, and parties
    3. Describe the suspicious activity pattern and supporting indicators
    4. Explain why the activity is considered suspicious based on available evidence
    5. Use clear, objective, regulatory-appropriate language
    6. Exclude assumptions, speculation, and facts not present in the case data
    """
    return {
        "reasoning_phase": [
            "Review risk analyst findings and classification",
            "Extract material facts, transactions, dates, amounts, and parties",
            "Identify evidence supporting the suspicious activity",
            "Organize facts into a clear chronological narrative",
            "Verify that all statements are supported by case evidence"
        ],
        "action_phase": [
            "Draft concise compliance narrative",
            "Include relevant dates, amounts, accounts, and parties",
            "Describe the suspicious activity pattern",
            "Explain why the activity is suspicious",
            "Use objective regulatory-appropriate language",
            "Do not introduce unsupported facts or speculation"
        ]
    }

def get_regulatory_requirements():
    """Key regulatory requirements for SAR narratives
    
    TODO: Use these requirements in your prompts:
    """
    return {
        "word_limit": 120,
        "required_elements": [
            "Customer identification",
            "Suspicious activity description", 
            "Transaction amounts and dates",
            "Why activity is suspicious"
        ],
        "terminology": [
            "Suspicious activity",
            "Regulatory threshold",
            "Financial institution",
            "Money laundering",
            "Bank Secrecy Act"
        ],
        "citations": [
            "31 CFR 1020.320 (BSA)",
            "12 CFR 21.11 (SAR Filing)",
            "FinCEN SAR Instructions"
        ]
    }

def validate_word_count(text: str, max_words: int = 120) -> bool:
    """Helper to validate word count
    
    """
    word_count = len(text.split())
    return word_count <= max_words

def get_completion(messages, client, model="gpt-4.1-mini") -> Any:
    """
    Function to get a completion from the OpenAI API.
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt  
        client: The OpenAI client instance
        model: The model to use (default is gpt-4o-mini)
    Returns:
        completion text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(
            f"LLM completion failed: {e}"
        ) from e

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
     
    print(f"📈 Loaded: {len(customers_data)} customers, {len(accounts_data)} accounts, {len(transactions_data)} transactions")
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

def test_narrative_generation(test_num):
    """Test the agent with a sample case
    Test the agent with a sample case
    
    This function is used to test the implementation:
    - Create sample case data
    - Initialize agent
    - Run analysis
    - Validate results

    """
    from src.risk_analyst_agent import RiskAnalystAgent

    pylogger.info("🧪 Testing Compliance Officer Agent")
    pylogger.info("Implementing test case")
    
    customers_data, accounts_data, transactions_data = testload_and_preprocess_data()

    selected_customer = select_customers(customers_data, accounts_data, transactions_data, select_n=test_num)
    pylogger.info(f"\n🔍 CUSTOMER SELECTED: {selected_customer['customer']['name']}")

    test_client, test_logger = create_client_and_logger()
    loader = DataLoader(test_logger)
    case_data_test = loader.create_case_from_data(
        selected_customer['customer'],
        selected_customer['accounts'], 
        selected_customer['transactions']
            )
    print("************* Sample case created successfully *************")

    risk_agent = RiskAnalystAgent(test_client, test_logger)
    compliance_agent = ComplianceOfficerAgent(test_client, test_logger)
    print("************* Agent initialized successfully *************")

    risk_analysis_result = risk_agent.analyze_case(case_data_test)
    compliance_output = compliance_agent.generate_compliance_narrative(case_data_test, risk_analysis_result)
    print("************* Analysis completed successfully *************")
    pprint (compliance_output)

    # Validating results
    print("************* Output Validation Started *************")
    if isinstance(compliance_output, ComplianceOfficerOutput):
        pylogger.info("Output conforms to ComplianceOfficerOutput")
        return compliance_output
    else:
        pylogger.error("Output is not a ComplianceOfficerOutput")
        return compliance_output


if __name__ == "__main__":
    print("✅ Compliance Officer Agent Module")
    print("ReACT prompting for regulatory narrative generation")
    print("\n📋 Testing Items:")
    print("• Design ReACT system prompt")
    print("• Implement generate_compliance_narrative method")
    print("• Add narrative validation (word count, terminology)")
    print("• Create regulatory citation system")
    print("• Test with sample risk analysis results")
    print("\n💡 Key Concepts:")
    print("• ReACT: Reasoning + Action structured prompting")
    print("• Regulatory Compliance: BSA/AML requirements")
    print("• Narrative Constraints: Word limits and terminology")
    print("• Audit Logging: Complete decision documentation")

    test_output = test_narrative_generation(4)
    print("\n========== TEST OUTPUT ==========")
    print(repr(test_output))
    print("=========================================\n")
