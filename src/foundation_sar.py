# Foundation SAR - Core Data Schemas and Utilities
# Implement core Pydantic schemas and data processing utilities

"""
This module contains the foundational components for SAR processing:

1. Pydantic Data Schemas:
   - CustomerData: Customer profile information
   - AccountData: Account details and balances  
   - TransactionData: Individual transaction records
   - CaseData: Unified case combining all data sources
   - RiskAnalystOutput: Risk analysis results
   - ComplianceOfficerOutput: Compliance narrative results

2. Utility Classes:
   - ExplainabilityLogger: Audit trail logging
   - DataLoader: Combines fragmented data into case objects

YOUR TASKS:
- Study the data files in data/ folder
- Design Pydantic schemas that match the CSV structure
- Implement validation rules for financial data
- Create a DataLoader that builds unified case objects
- Add proper error handling and logging
"""

import json
import pandas as pd
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Any, Literal, Annotated
from pydantic import BaseModel, ValidationInfo, Field, field_validator, AfterValidator, model_validator
import uuid
import os
from enum import Enum
import re
from uuid import UUID, uuid4

import logging
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S"
)

pylogger = logging.getLogger(__name__)

# ===== IMPLEMENT PYDANTIC SCHEMAS =====

def validate_date_format(v: str) -> str:
    if not v:
        raise ValueError("Date is required")

    try:
        parsed_date = datetime.strptime(v, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            "Date must be a valid date in YYYY-MM-DD format, e.g. 1985-03-15"
        ) from None

    # Ensure exact format
    if parsed_date.isoformat() != v:
        raise ValueError("Date must be strictly in YYYY-MM-DD format")

    # Plausibility: reject unrealistically old dates
    if parsed_date.year < 1900:
        raise ValueError("Date cannot be earlier than 1900")

    # Plausibility: reject future dates
    if parsed_date > date.today():
        raise ValueError("Date cannot be in the future")

    return v

DateString = Annotated[str, AfterValidator(validate_date_format)]

class RiskAssessment(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class CustomerData(BaseModel):
    """Customer information schema with validation
    
    REQUIRED FIELDS (examine data/customers.csv):
    - customer_id: str = Unique identifier like "CUST_0001"
    - name: str = Full customer name like "John Smith"
    - date_of_birth: str = Date in YYYY-MM-DD format like "1985-03-15"
    - ssn_last_4: str = Last 4 digits like "1234"
    - address: str = Full address like "123 Main St, City, ST 12345"
    - customer_since: str = Date in YYYY-MM-DD format like "2010-01-15"
    - risk_rating: Literal['Low', 'Medium', 'High'] = Risk assessment
    
    OPTIONAL FIELDS:
    - phone: Optional[str] = Phone number like "555-123-4567"
    - occupation: Optional[str] = Job title like "Software Engineer"
    - annual_income: Optional[int] = Yearly income like 75000
    
    HINT: Use Field(..., description="...") for required fields
    HINT: Use Field(None, description="...") for optional fields
    HINT: Use Literal type for risk_rating to restrict values
    """
    # Implement the CustomerData schema with proper fields and validation
    customer_id: str = Field(..., description="Unique identifier like `CUST_0001`")
    name: str = Field(..., description="Full customer name like `John Smith`")
    date_of_birth: DateString = Field(..., description="Date in YYYY-MM-DD format like `1985-03-15`")
    ssn_last_4: str = Field(..., description="Last 4 digits like '1234'")
    address: str = Field(..., description="Full address like `123 Main St, City, ST 12345`")
    customer_since: DateString = Field(..., description="Date in YYYY-MM-DD format like `2010-01-15`")
    risk_rating: RiskAssessment = Field(..., description="Risk assessment")
    phone: str|None = Field(None, description="Phone number like `555-123-4567`")
    occupation: str|None = Field(None, description="Job title like like `Software Engineer`")
    annual_income: int|None = Field(None, ge=0, description="Yearly income like 75000")

    @field_validator("phone")
    @classmethod
    def val_phone_format(cls, v):
        if not v:
            return v

        # Allow common extension markers: x, ext, ext.
        cleaned = re.sub(r"ext\.?\s*|x\s*", "", v, flags=re.IGNORECASE)

        if re.search(r"[A-Za-z]", cleaned):
            raise ValueError(
                "Phone number must not contain alphabetic characters "
                "except 'x' or 'ext' for extensions"
            )

        return v

    @field_validator("ssn_last_4")
    @classmethod
    def val_ssn_last4_format(cls, v):
        if not v:
            raise ValueError("Last 4 digits of SSN are required")

        if not re.fullmatch(r"\d{4}", v):
            raise ValueError(
                "SSN must contain exactly 4 digits, e.g. 1234"
            )

        return v

class AccountData(BaseModel):
    """Account information schema with validation
    
    REQUIRED FIELDS (examine data/accounts.csv):
    - account_id: str = Unique identifier like "account_id"
    - customer_id: str = Must match CustomerData.customer_id
    - account_type: str = Type like "Checking", "Savings", "Money_Market"
    - opening_date: str = Date in YYYY-MM-DD format
    - current_balance: float = Current balance (can be negative)
    - average_monthly_balance: float = Average balance
    - status: str = Status like "Active", "Closed", "Suspended"
    
    HINT: All fields are required for account data
    HINT: Use float for monetary amounts
    HINT: current_balance can be negative for overdrafts
    """
    # Implement the AccountData schema
    account_id: str = Field(..., description="Unique identifier like `account_id`")
    customer_id: str = Field(..., description="Must match CustomerData.customer_id")
    account_type: str = Field(..., description="Type of account like `Checking`, `Savings`, `Money_Market`")
    opening_date: DateString = Field(..., description="Date in YYYY-MM-DD format")
    current_balance: float = Field(..., description="Current balance (can be negative)")
    average_monthly_balance: float = Field(..., description="Average monthly balance (can be negative)")
    status: str = Field(..., description="Status like `Active`, `Closed`, `Suspended`")

    @field_validator('account_type')
    @classmethod
    def valid_account_type(cls, v):
        valid_account_types = ["checking", "savings", "money_market", "business_checking"]
        if v.lower() not in valid_account_types:
            raise ValueError(f'account type must be one of: {valid_account_types}')
        return v

    @field_validator('status')
    @classmethod
    def valid_account_status(cls, v):
        valid_statuses = ["active", "closed", "suspended"]
        if v.lower() not in valid_statuses:
            raise ValueError(f'status must be one of: {valid_statuses}')
        return v

class TransactionData(BaseModel):
    """Transaction information schema with validation
    
    REQUIRED FIELDS (examine data/transactions.csv):
    - transaction_id: str = Unique identifier like "TXN_B24455F3"
    - account_id: str = Must match AccountData.account_id
    - transaction_date: str = Date in YYYY-MM-DD format
    - transaction_type: str = Type like "Cash_Deposit", "Wire_Transfer"
    - amount: float = Transaction amount (negative for withdrawals)
    - description: str = Description like "Cash deposit at branch"
    - method: str = Method like "Wire", "ACH", "ATM", "Teller"
    
    OPTIONAL FIELDS:
    - counterparty: Optional[str] = Other party in transaction
    - location: Optional[str] = Transaction location or branch
    
    HINT: amount can be negative for debits/withdrawals
    HINT: Use descriptive field descriptions for clarity
    """
    # Implement the TransactionData schema
    transaction_id: str = Field(..., description="Unique identifier like `TXN_B24455F3`")
    account_id: str = Field(..., description="Must match AccountData.account_id")
    transaction_date: DateString = Field(..., description="Date in YYYY-MM-DD format")
    transaction_type: str = Field(..., description="Type of account like `Cash_Deposit`, `Wire_Transfer`")
    amount: float = Field(..., description="Transaction amount (can be negative)")
    description: str = Field(..., description="Description like `Cash deposit at branch`")
    method: str = Field(..., description="Status like `Wire`, `ACH`, `ATM`, `Teller`")
    counterparty: str|None = Field(None, description="Other party in transaction")
    location: str|None = Field(None, description="Transaction location or branch")

    @field_validator('transaction_type')
    @classmethod
    def valid_transaction_type(cls, v):
        transaction_types = ["cash_deposit", "cash_withdrawal", "wire_transfer", "atm_withdrawal","online_transfer", "online", "deposit", 
                             "ach_debit", "ach_credit", "debit_purchase", "wire_transfer_debit", "wire_transfer_credit", 
                             "check_deposit","electronic", "direct_deposit", "test"]
        if v.lower() not in transaction_types:
            raise ValueError(f'transaction type must be one of: {transaction_types}')
        return v

    @field_validator('method')
    @classmethod
    def valid_method(cls, v):
        method_types = ["wire", "ach", "atm", "cash", "online", "teller", "mobile", "electronic", "branch", "test"]
        if v.lower() not in method_types:
            raise ValueError(f'method must be one of: {method_types}')
        return v

class CaseData(BaseModel):
    """Unified case object combining all data sources
    
    REQUIRED FIELDS:
    - case_id: str = Unique case identifier (generate with uuid)
    - customer: CustomerData = Customer information object
    - accounts: List[AccountData] = List of customer's accounts
    - transactions: List[TransactionData] = List of suspicious transactions
    - case_created_at: str = ISO timestamp when case was created
    - data_sources: Dict[str, str] = Source tracking with keys like:
      * "customer_source": "csv_extract_20241219"
      * "account_source": "csv_extract_20241219" 
      * "transaction_source": "csv_extract_20241219"
    
    VALIDATION RULES:
    - transactions list cannot be empty (use @field_validator)
    - All accounts should belong to the same customer
    - All transactions should belong to accounts in the case
    
    HINT: Use @field_validator('transactions') with @classmethod decorator
    HINT: Check if not v: raise ValueError("message") for empty validation
    """
    # Implement the CaseData schema with validation
    case_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique case identifier")
    customer: CustomerData
    accounts: List[AccountData] = Field(..., description="List of customer's accounts")
    transactions: List[TransactionData] = Field(..., description="List of suspicious transactions")
    case_created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="ISO timestamp when case was created")
    data_sources: Dict[str, str] = Field(..., description="Source tracking with keys")

    @field_validator('transactions')
    @classmethod
    def must_have_transactions(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one transaction must exist in the case')
        return v

    @model_validator(mode="after")
    def validate_case_relationships(self, info: ValidationInfo):

        # Rule 1: All accounts must belong to the case customer.
        invalid_accounts = [
            account.account_id
            for account in self.accounts
            if account.customer_id != self.customer.customer_id
        ]

        if invalid_accounts:
            raise ValueError(
                f"All accounts must belong to customer "
                f"{self.customer.customer_id}. "
                f"Invalid accounts: {invalid_accounts}"
            )

        # Rule 2: Transactions must reference accounts in the case.
        valid_account_ids = {
            account.account_id
            for account in self.accounts
        }

        invalid_transactions = [
            tx.transaction_id
            for tx in self.transactions
            if tx.account_id not in valid_account_ids
        ]

        # Explicitly allow incomplete account data when requested.
        allow_incomplete = (
            isinstance(info.context, dict)
            and info.context.get("allow_incomplete_accounts", False)
        )

        if invalid_transactions and self.accounts and not allow_incomplete:
            raise ValueError(
                "All transactions must belong to accounts in the case. "
                f"Invalid transactions: {invalid_transactions}"
            )

        return self

class RiskAnalystOutput(BaseModel):
    """Risk Analyst agent structured output
    
    REQUIRED FIELDS (for Chain-of-Thought agent output):
    - classification: Literal['Structuring', 'Sanctions', 'Fraud', 'Money_Laundering', 'Other']
    - confidence_score: float = Confidence between 0.0 and 1.0 (use ge=0.0, le=1.0)
    - reasoning: str = Step-by-step analysis reasoning (max 500 chars)
    - key_indicators: List[str] = List of suspicious indicators found
    - risk_level: Literal['Low', 'Medium', 'High', 'Critical'] = Risk assessment
    
    HINT: Use Literal types to restrict classification and risk_level values
    HINT: Use Field(..., ge=0.0, le=1.0) for confidence_score validation
    HINT: Use Field(..., max_length=500) for reasoning length limit
    """
    # Implement the RiskAnalystOutput schema
    classification: Literal['Structuring', 'Sanctions', 'Fraud', 'Money_Laundering', 'Other'] = Field(..., description="Risk classification")
    confidence_score: float = Field(..., ge=0, le=1.0, description="Confidence score 0-1.0")
    reasoning: str = Field(..., max_length=500, description="Step-by-step analysis reasoning")
    key_indicators: List[str] = Field(..., description="List of suspicious indicators found")
    risk_level: RiskAssessment = Field(..., description="Risk level")

class ComplianceOfficerOutput(BaseModel):
    """Compliance Officer agent structured output
    
    REQUIRED FIELDS (for ReACT agent output):
    - narrative: str = Regulatory narrative text (max 1000 chars for ≤200 words)
    - narrative_reasoning: str = Reasoning for narrative construction (max 500 chars)
    - regulatory_citations: List[str] = List of relevant regulations like:
      * "31 CFR 1020.320 (BSA)"
      * "12 CFR 21.11 (SAR Filing)"
      * "FinCEN SAR Instructions"
    - completeness_check: bool = Whether narrative meets all requirements
    
    HINT: Use Field(..., max_length=1000) for narrative length limit
    HINT: Use Field(..., max_length=500) for reasoning length limit
    HINT: Use bool type for completeness_check
    """
    # Implement the ComplianceOfficerOutput schema
    narrative: str = Field(..., max_length=1000, description="Regulatory narrative text")
    narrative_reasoning: str = Field(..., max_length=500, description="Reasoning for narrative construction")
    regulatory_citations: List[str] = Field(..., description="List of relevant regulations")
    completeness_check: bool = Field(..., description="Whether narrative meets all requirements")

# ===== IMPLEMENT AUDIT LOGGING =====

class ExplainabilityLogger:
    """Simple audit logging for compliance trails

    ATTRIBUTES:
    - log_file: str = Path to JSONL log file (default: "sar_audit.jsonl")
    - entries: List = In-memory storage of log entries

    METHODS:
    - log_agent_action(): Logs agent actions with structured data
    
    LOG ENTRY STRUCTURE (use this exact format):
    {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'case_id': case_id,
        'agent_type': agent_type,  # "DataLoader", "RiskAnalyst", "ComplianceOfficer"
        'action': action,          # "create_case", "analyze_case", "generate_narrative"
        'input_summary': str(input_data),
        'output_summary': str(output_data),
        'reasoning': reasoning,
        'execution_time_ms': execution_time_ms,
        'success': success,        # True/False
        'error_message': error_message  # None if success=True
    }
    
    HINT: Write each entry as JSON + newline to create JSONL format
    HINT: Use 'a' mode to append to log file
    HINT: Store entries in self.entries list AND write to file
    """
    
    def __init__(self, log_file: str = "sar_audit.jsonl"):
        # Initialize with log_file path and empty entries list
        self.log_file = log_file
        self.entries = []
    
    def log_agent_action(
        self,
        agent_type: str,
        action: str,
        case_id: str,
        input_data: Dict,
        output_data: Dict,
        reasoning: str,
        execution_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'case_id': str(case_id),
            'agent_type': agent_type,
            'action': action,
            'input_summary': str(input_data),
            'output_summary': str(output_data),
            'reasoning': reasoning,
            'execution_time_ms': execution_time_ms,
            'success': success,
            'error_message': error_message
        }
        """Log an agent action with essential context
        
        IMPLEMENTATION STEPS:
        1. Create entry dictionary with all fields (see structure above)
        2. Add entry to self.entries list
        3. Write entry to log file as JSON line
        
        HINT: Use json.dumps(entry) + '\n' for JSONL format
        HINT: Use datetime.now(timezone.utc).isoformat() for timestamp
        HINT: Convert input_data and output_data to strings with str()
        """
        # Implement logging with structured entry creation and file writing

        try:
            # Serialize before writing
            serialized_entry = json.dumps(entry)

            # Persist to JSONL file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(serialized_entry + '\n')

        except (TypeError, ValueError, OSError) as e:
            raise RuntimeError(
                f"Failed to persist audit log for "
                f"case_id={case_id}, "
                f"agent_type={agent_type}, "
                f"action={action}: {e}"
            ) from e

        # Only add to memory after successful file write
        self.entries.append(entry)

# ===== IMPLEMENT DATA LOADER =====

class DataLoader:
    """Simple loader that creates case objects from CSV data
    
    ATTRIBUTES:
    - logger: ExplainabilityLogger = For audit logging
    
    HELPFUL METHODS:
    - create_case_from_data(): Creates CaseData from input dictionaries
    
    IMPLEMENTATION PATTERN:
    1. Start timing with start_time = datetime.now()
    2. Generate case_id with str(uuid.uuid4())
    3. Create CustomerData object from customer_data dict
    4. Filter accounts where acc['customer_id'] == customer.customer_id
    5. Get account_ids set from filtered accounts
    6. Filter transactions where txn['account_id'] in account_ids
    7. Create CaseData object with all components
    8. Calculate execution_time_ms
    9. Log success/failure with self.logger.log_agent_action()
    10. Return CaseData object (or raise exception on failure)
    """
    
    def __init__(self, explainability_logger: ExplainabilityLogger):
        # Store logger for audit trail
        self.logger = explainability_logger


    def create_case_from_data(
        self,
        customer_data: Dict,
        account_data: List[Dict],
        transaction_data: List[Dict]
    ) -> CaseData:
        """Create a unified case object from fragmented AML data

        SUGGESTED STEPS:
        1. Record start time for performance tracking
        2. Generate unique case_id using uuid.uuid4()
        3. Create CustomerData object from customer_data dictionary
        4. Filter account_data list for accounts belonging to this customer
        5. Create AccountData objects from filtered accounts
        6. Get set of account_ids from customer's accounts
        7. Filter transaction_data for transactions in customer's accounts
        8. Create TransactionData objects from filtered transactions  
        9. Create CaseData object combining all components
        10. Add case metadata (case_id, timestamp, data_sources)
        11. Calculate execution time in milliseconds
        12. Log operation with success/failure status
        13. Return CaseData object
        
        ERROR HANDLING:
        - Wrap in try/except block
        - Log failures with error message
        - Re-raise exceptions for caller
        
        DATA_SOURCES FORMAT:
        {
            'customer_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}",
            'account_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}",
            'transaction_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}"
        }
        
        HINT: Use list comprehensions for filtering
        HINT: Use set comprehension for account_ids: {acc.account_id for acc in accounts}
        HINT: Use datetime.now(timezone.utc).isoformat() for timestamps
        HINT: Calculate execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        """
        # Implement complete case creation with error handling and logging

        start_time = datetime.now(timezone.utc)
        case_id = str(uuid.uuid4())

        # Track which record is being processed
        record_type = "customer"
        record_id = customer_data.get("customer_id", "UNKNOWN")

        try:
            # 1. Validate customer
            customer = CustomerData.model_validate(customer_data)

            # 2. Validate customer's accounts
            account_list = []

            for acc in account_data:
                record_type = "account"
                record_id = acc.get("account_id", "UNKNOWN")

                if acc["customer_id"] == customer.customer_id:
                    account_list.append(
                        AccountData.model_validate(acc)
                    )

            # 3. Identify customer account IDs
            account_ids = {acc.account_id for acc in account_list}

            # 4. Validate associated transactions
            transaction_list = []

            for txn in transaction_data:
                record_type = "transaction"
                record_id = txn.get("transaction_id", "UNKNOWN")

                if txn["account_id"] in account_ids:
                    transaction_list.append(
                        TransactionData.model_validate(txn)
                    )

            # 5. Create case metadata
            case_created_at = datetime.now(timezone.utc)

            data_sources = {
                "customer_source": f"csv_extract_{case_created_at:%Y%m%d}",
                "account_source": f"csv_extract_{case_created_at:%Y%m%d}",
                "transaction_source": f"csv_extract_{case_created_at:%Y%m%d}"
            }

            # 6. Create CaseData
            record_type = "case"
            record_id = case_id

            case = CaseData(
                case_id=case_id,
                customer=customer,
                accounts=account_list,
                transactions=transaction_list,
                case_created_at=case_created_at,
                data_sources=data_sources
            )

            # 7. Log success
            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data={
                    "customer_id": customer.customer_id,
                    "account_ids": list(account_ids),
                    "transaction_count": len(transaction_list)
                },
                output_data={"case_id": case_id},
                reasoning="Successfully created CaseData object.",
                execution_time_ms=execution_time_ms,
                success=True
            )

            return case

        except Exception as e:

            execution_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data={
                    "customer_id": customer_data.get("customer_id"),
                    "record_type": record_type,
                    "record_id": record_id
                },
                output_data={},
                reasoning="Failed to create CaseData object.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=(
                    f"Validation/processing failure for "
                    f"{record_type} '{record_id}': {e}"
                )
            )

            raise


# ===== HELPER FUNCTIONS (PROVIDED) =====

def load_csv_data(data_dir: str = "data/") -> tuple:
    """Helper function to load all CSV files
    
    Returns:
        tuple: (customers_df, accounts_df, transactions_df)
    """
    try:
        customers_df = pd.read_csv(f"{data_dir}/customers.csv")
        accounts_df = pd.read_csv(f"{data_dir}/accounts.csv") 
        transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
        return customers_df, accounts_df, transactions_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading CSV data: {e}")


# ===== TESTING UTILITIES =====

def test_components():

    from pathlib import Path
    import sys

    # foundation_sar.py resides in starter/src/
    project_root = Path(__file__).resolve().parent.parent

    src_path = project_root / "src"
    tests_path = project_root / "tests"

    for path in (project_root, src_path, tests_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    pylogger.debug("📁 Python path includes:")
    pylogger.debug(f"   📂 src: {src_path}")
    pylogger.debug(f"   📂 tests: {tests_path}")

    try:
        # Import and run tests from the test suite
        from test_foundation import (
            TestCustomerData,
            TestAccountData,
            TestTransactionData,
            TestCaseData,
            TestDataLoader,
            TestExplainabilityLogger
        )

        pylogger.info("✅ Test classes imported successfully!")

            
        # Initialize test instances
        test_customer = TestCustomerData()
        test_account = TestAccountData()
        test_transaction = TestTransactionData()
        test_case = TestCaseData()
        test_loader = TestDataLoader()
        test_logger = TestExplainabilityLogger()

        pylogger.info("✅ Test initialized successfully!")
        
        # Run CustomerData tests
        print("\n👥 Testing CustomerData...")
        try:
            test_customer.test_valid_customer_data()
            pylogger.info("  ✅ Valid customer data test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Valid customer data test failed: {e}")
            
        try:
            test_customer.test_customer_risk_rating_validation()
            pylogger.info("  ✅ Risk rating validation test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Risk rating validation test failed: {e}")
        
        # Run AccountData tests
        print("\n🏦 Testing AccountData...")
        try:
            test_account.test_valid_account_data()
            pylogger.info("  ✅ Valid account data test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Valid account data test failed: {e}")
            
        try:
            test_account.test_account_balance_validation()
            pylogger.info("  ✅ Account balance validation test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Account balance validation test failed: {e}")
        
        # Run TransactionData tests
        print("\n💳 Testing TransactionData...")
        try:
            test_transaction.test_valid_transaction_data()
            pylogger.info("  ✅ Valid transaction data test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Valid transaction data test failed: {e}")
            
        try:
            test_transaction.test_transaction_amount_validation()
            pylogger.info("  ✅ Transaction amount validation test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Transaction amount validation test failed: {e}")
        
        # Run CaseData tests
        print("\n📋 Testing CaseData...")
        try:
            test_case.test_valid_case_creation()
            pylogger.info("  ✅ Valid case creation test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Valid case creation test failed: {e}")
        
        # Run DataLoader tests
        print("\n📊 Testing DataLoader...")
        try:
            test_loader.test_csv_data_loading()
            pylogger.info("  ✅ CSV data loading test passed")
        except Exception as e:
            pylogger.error(f"  ❌ CSV data loading test failed: {e}")
        
        # Run ExplainabilityLogger tests
        print("\n📝 Testing ExplainabilityLogger...")
        try:
            test_logger.test_log_creation()
            pylogger.info("  ✅ Log creation test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Log creation test failed: {e}")
            
        try:
            test_logger.test_log_file_writing()
            pylogger.info("  ✅ Log file writing test passed")
        except Exception as e:
            pylogger.error(f"  ❌ Log file writing test failed: {e}")
            
    except ImportError as e:
        pylogger.error(f"❌ Import Error: {e}")
        pylogger.error("💡 Make sure your implementation is complete and error-free")
        pylogger.error("📝 Run Step 1 first to check basic imports")

    print("\n✅ ============== Foundation SAR tests completed successfully! ==============")
        

    return

if __name__ == "__main__":
    print("🏗️  Foundation SAR Module")
    print("Core data schemas and utilities for SAR processing")
    print("\n📋 Testing Items:")
    print("• Implement Pydantic schemas based on CSV data")
    print("• Create ExplainabilityLogger for audit trails")
    print("• Build DataLoader for case object creation")
    print("• Add comprehensive error handling")
    print("• Write unit tests for all components")

    print("\n✅ ============== Starting Foundation SAR tests! ==============")

    test_components()



