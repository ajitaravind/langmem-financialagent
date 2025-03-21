"""
Financial Document Analysis Agent with Episodic Memory
"""

# Load environment variables

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from pydantic import BaseModel, Field
from typing import Dict, Literal, TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

import uuid
from datetime import datetime
from langmem import create_manage_memory_tool, create_search_memory_tool


# Initialize LLM and Memory Store
llm = init_chat_model("openai:gpt-4o")
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Memory tools:
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "financial_assistant", 
        "default_user",  
        "collection"
    ),
    store=store  
)

search_memory_tool = create_search_memory_tool(
    namespace=(
        "financial_assistant",
        "default_user",  
        "collection"
    ),
    store=store 
)

# Define Models
class FinancialDocument(BaseModel):
    """Model for financial documents and transactions"""
    doc_type: Literal["transaction", "compliance", "information", "query"]
    risk_level: Literal["low", "high"]
    content: str
    metadata: Dict = Field(default_factory=dict)

class Router(BaseModel):
    """Analyze financial documents and route according to content"""
    reasoning: str = Field(description="Step-by-step reasoning behind the classification")
    classification: Literal["ignore", "respond"] = Field(
        description="Classification of the document: 'ignore' for routine documents, "
        "'respond' for high-priority items"
    )

# Initialize Router
llm_router = llm.with_structured_output(Router)

# Memory Management Functions
def store_document_decision(doc: dict, classification: str, reasoning: str):
    """Store document processing decisions in episodic memory"""
    # Calculate risk factors
    risk_factors = []
    if doc.get("amount", 0) >= 10000:
        risk_factors.append("high_value")
    if doc.get("type", "").lower() == "international":
        risk_factors.append("international")
    if doc.get("risk_level", "").lower() == "high":
        risk_factors.append("high_risk")
        
    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "document": {
            "type": doc["doc_type"],
            "risk_level": doc["risk_level"],
            "amount": doc.get("amount"),
            "transaction_type": doc.get("type")
        },
        "analysis": {
            "classification": classification,
            "reasoning": reasoning,
            "risk_factors": risk_factors,
            "requires_reporting": doc.get("amount", 0) >= 10000
        },
        "metadata": doc.get("metadata", {})
    }
    
    store.put(
        ("financial_assistant", "processing", "decisions"),
        str(uuid.uuid4()),
        memory_entry
    )

def get_similar_decisions(doc: dict, limit: int = 2) -> str:
    """Retrieve similar past decisions from episodic memory"""
    results = store.search(
        ("financial_assistant", "processing", "decisions"),
        query=str({"document": {
            "type": doc["doc_type"],
            "risk_level": doc["risk_level"],
            "amount": doc.get("amount"),
            "transaction_type": doc.get("type")
        }}),
        limit=limit
    )
    
    if not results:
        return "No similar past decisions found."
    
    similar_cases = []
    for item in results:
        case = f"""Previous Case ({item.value['timestamp']}):
- Document Type: {item.value['document']['type']}
- Risk Level: {item.value['document']['risk_level']}
- Amount: ${item.value['document'].get('amount', 'N/A')}
- Decision: {item.value['analysis']['classification']}
- Reasoning: {item.value['analysis']['reasoning']}
- Risk Factors: {', '.join(item.value['analysis']['risk_factors'])}"""
        similar_cases.append(case)
    
    return "\n\n".join(similar_cases)

# Add a new function to detect patterns
def detect_transaction_patterns(doc: dict) -> str:
    """Detect patterns in recent transactions"""
    recent_transactions = store.search(
        ("financial_assistant", "processing", "decisions"),
        query=None,  # Get all recent decisions
        limit=5
    )
    
    if not recent_transactions:
        return "No transaction history available for pattern detection."
        
    patterns = {
        "high_value_count": 0,
        "international_count": 0,
        "high_risk_count": 0,
        "total_amount": 0
    }
    
    for transaction in recent_transactions:
        if transaction.value["document"].get("amount", 0) >= 10000:
            patterns["high_value_count"] += 1
        if transaction.value["document"].get("transaction_type", "").lower() == "international":
            patterns["international_count"] += 1
        if transaction.value["document"].get("risk_level", "").lower() == "high":
            patterns["high_risk_count"] += 1
        patterns["total_amount"] += transaction.value["document"].get("amount", 0)
    
    pattern_summary = []
    if patterns["high_value_count"] >= 2:
        pattern_summary.append(f"Multiple high-value transactions detected ({patterns['high_value_count']} in recent history)")
    if patterns["international_count"] >= 2:
        pattern_summary.append(f"Multiple international transactions detected ({patterns['international_count']} in recent history)")
    if patterns["high_risk_count"] >= 2:
        pattern_summary.append(f"Multiple high-risk transactions detected ({patterns['high_risk_count']} in recent history)")
    
    if not pattern_summary:
        return "No significant patterns detected in recent transactions."
    
    return "Pattern Analysis:\n- " + "\n- ".join(pattern_summary)

# Tools
@tool
def analyze_transaction(
    amount: float,
    transaction_type: str,
    transaction_id: str,
    risk_level: str
) -> str:
    """Analyze a financial transaction."""
    findings = []
    
    # Amount threshold analysis
    if amount >= 10000:
        findings.append("Requires mandatory reporting due to amount exceeding $10,000")
    elif amount >= 5000:
        findings.append("Large transaction - additional review recommended")
    
    # Transaction type analysis
    type_risks = {
        "international": "International transaction - verify compliance with cross-border regulations",
        "wire": "Wire transfer - verify sender/recipient details",
        "crypto": "Cryptocurrency transaction - ensure compliance with digital asset regulations",
        "cash": "Cash transaction - verify source of funds"
    }
    if transaction_type.lower() in type_risks:
        findings.append(type_risks[transaction_type.lower()])
    
    # Risk level analysis
    risk_actions = {
        "high": "Immediate review required - escalate to compliance team",
        "critical": "URGENT: Transaction held for compliance officer review",
        "medium": "Standard review process required",
        "low": "Routine processing approved"
    }
    if risk_level.lower() in risk_actions:
        findings.append(risk_actions[risk_level.lower()])
    
    # Combine findings
    analysis_summary = f"Transaction {transaction_id} Analysis:\n"
    analysis_summary += f"- Type: {transaction_type} for ${amount:,.2f}\n"
    analysis_summary += f"- Risk Level: {risk_level}\n"
    analysis_summary += "\nFindings:\n"
    for i, finding in enumerate(findings, 1):
        analysis_summary += f"{i}. {finding}\n"
    
    return analysis_summary

# Update the create_prompt function
def create_prompt(state):
    return [
        {
            "role": "system",
            "content": """You are a Financial Document Analyzer with both episodic and semantic memory.
Use the analyze_transaction tool to analyze financial transactions.
Use manage_memory to store important financial rules and patterns.
Use search_memory to retrieve relevant past knowledge.

Available Tools:
1. analyze_transaction - Analyze financial transactions
2. manage_memory - Store important financial rules and patterns
3. search_memory - Retrieve relevant past knowledge

Base your decisions on:
1. Transaction details
2. Historical patterns
3. Stored financial rules
4. Regulatory requirements"""
        }
    ] + state['messages']

# Update the tools list in the agent creation
tools = [
    analyze_transaction,
    manage_memory_tool,
    search_memory_tool
]

# Update the agent creation
agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
    store=store
)

# State Management
class State(TypedDict):
    document_input: Dict
    messages: Annotated[list, add_messages]

def process_document(state: State) -> Command:
    """Process and classify financial documents with memory"""
    doc = state['document_input']
    
    # Create document object
    financial_doc = FinancialDocument(
        doc_type=doc["doc_type"],
        risk_level=doc["risk_level"],
        content=doc["content"],
        metadata=doc.get("metadata", {})
    )
    
    # Handle information and query documents differently
    if doc["doc_type"] in ["information", "query"]:
        return Command(goto="analysis_agent", update={
            "messages": [
                {
                    "role": "system",
                    "content": """You are a Financial Document Analyzer with semantic memory capabilities.
For information documents:
- Use manage_memory with action='create' to store new information
- Do not specify an ID, it will be generated automatically
- Confirm what was stored

For query documents:
- Use search_memory to find relevant information
- Provide a clear, direct answer based on the stored information
- If no information is found, clearly state that"""
                },
                {
                    "role": "user",
                    "content": f"""Please handle this {doc['doc_type']}:
Content: {doc['content']}
Type: {doc['type']}

If this is information, please store it in semantic memory.
If this is a query, please search semantic memory to answer it."""
                }
            ]
        })
    
    # For transaction and compliance documents, check risk rules
    similar_decisions = get_similar_decisions(doc)
    patterns = detect_transaction_patterns(doc)
    
    # Get applicable risk rules
    risk_rules = store.search(
        ("financial_assistant", "default_user", "collection"),
        query="Risk threshold rule",
        limit=5
    )
    
    triggered_rules = []
    for rule in risk_rules:
        rule_content = rule.value["content"].lower()
        # Check international transaction threshold
        if (
            "international" in rule_content 
            and doc.get("type", "").lower() == "international"
            and "$10,000" in rule_content 
            and doc.get("amount", 0) > 10000
        ):
            triggered_rules.append("Enhanced due diligence required - International transaction over $10,000")
            
        # Check multiple transactions threshold
        if (
            "multiple transactions" in rule_content 
            and "24 hours" in rule_content
            and patterns.find("Multiple high-value transactions") != -1
        ):
            triggered_rules.append("Compliance review required - Multiple high-value transactions in 24 hours")
            
        # Check international transaction frequency
        if (
            "three or more international" in rule_content
            and doc.get("type", "").lower() == "international"
            and patterns.find("Multiple international transactions") != -1
        ):
            triggered_rules.append("Special monitoring required - High frequency of international transactions")
    
    # Enhanced system prompt with memory, patterns, and triggered rules
    system_prompt = """
    You are a financial document analyzer. Classify documents as either:
    - IGNORE: Regular transactions and routine documents
    - RESPOND: High-value transactions, suspicious patterns, or high-risk documents
    
    Consider these similar past decisions when making your classification:
    {similar_decisions}
    
    Transaction Pattern Analysis:
    {patterns}
    
    Triggered Risk Rules:
    {triggered_rules}
    
    Base your decision on:
    1. Transaction amount and type
    2. Risk level
    3. Historical patterns from similar cases
    4. Triggered risk rules
    5. Regulatory requirements
    """
    
    user_prompt = f"""
    Document Type: {financial_doc.doc_type}
    Risk Level: {financial_doc.risk_level}
    Content: {financial_doc.content}
    Amount: ${doc.get('amount', 'N/A')}
    Transaction Type: {doc.get('type', 'N/A')}
    """
    
    # Get classification
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt.format(
                similar_decisions=similar_decisions,
                patterns=patterns,
                triggered_rules="\n".join(triggered_rules) if triggered_rules else "No risk rules triggered"
            )},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    # Store the decision
    store_document_decision(doc, result.classification, result.reasoning)
    
    # Process based on classification
    if result.classification == "respond":
        return Command(goto="analysis_agent", update={
            "messages": [
                {
                    "role": "user",
                    "content": f"""Please analyze this transaction:
Amount: ${doc['amount']}
Type: {doc['type']}
ID: {doc['metadata']['transaction_id']}
Risk Level: {doc['risk_level']}

Similar past decisions for reference:
{similar_decisions}

Pattern Analysis:
{patterns}

Triggered Risk Rules:
{chr(10).join(triggered_rules) if triggered_rules else "No risk rules triggered"}"""
                }
            ]
        })
    
    return Command(goto=END, update={
        "messages": [
            {
                "role": "assistant",
                "content": f"""Transaction {doc['metadata']['transaction_id']} has been classified as routine and requires no further analysis:
- Type: {doc['type']}
- Amount: ${doc['amount']:,.2f}
- Risk Level: {doc['risk_level']}
- Content: {doc['content']}

This decision was made considering:
1. Similar past cases:
{similar_decisions}

2. Transaction Patterns:
{patterns}

3. Risk Rules Review:
{chr(10).join(triggered_rules) if triggered_rules else "No risk rules triggered"}

This transaction can be processed through standard channels."""
            }
        ]
    })

# Add after the process_document function:

def print_all_memories():
    """Print both episodic and semantic memories"""
    print("\n=== Episodic Memories (Transaction Decisions) ===")
    examples = store.search(
        ('financial_assistant', 'processing', 'decisions'),
        query=None,
        limit=10
    )
    for item in examples:
        print(f"\nTransaction Type: {item.value['document']['type']}")
        print(f"Risk Level: {item.value['document']['risk_level']}")
        print(f"Amount: ${item.value['document'].get('amount', 'N/A')}")
        print(f"Classification: {item.value['analysis']['classification']}")
        print(f"Risk Factors: {', '.join(item.value['analysis']['risk_factors'])}")

    print("\n=== Semantic Memories (Financial Rules & Patterns) ===")
    facts = store.search(
        ('financial_assistant', 'default_user', 'collection'),
        query=None,
        limit=10
    )
    for item in facts:
        print(f"\nFact: {item.value['content']}")

# Initialize Graph
financial_agent = StateGraph(State)
financial_agent = financial_agent.add_node("document_processor", process_document)
financial_agent = financial_agent.add_node("analysis_agent", agent)
financial_agent = financial_agent.add_edge(START, "document_processor")
financial_agent = financial_agent.compile()

# Update the main section to include configuration
if __name__ == "__main__":
    config = {
        "langgraph_user_id": "default_user"
    }
    
    # Store risk threshold rules
    print("\nStoring risk threshold rules:")
    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "information",
                "risk_level": "low",
                "content": "Risk threshold rule: Any international transaction above $10,000 requires enhanced due diligence",
                "type": "risk_rule",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "information",
                "risk_level": "low",
                "content": "Risk threshold rule: Multiple transactions totaling over $25,000 in 24 hours require compliance review",
                "type": "risk_rule",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "information",
                "risk_level": "low",
                "content": "Risk threshold rule: Three or more international transactions within 24 hours require special monitoring",
                "type": "risk_rule",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    # First, let's store some financial information
    print("\nStoring exchange rate information:")
    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "information",
                "risk_level": "low",
                "content": "The current exchange rate is: 1 USD = 0.82 EUR",
                "type": "exchange_rate",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    # Store another piece of information
    print("\nStoring another exchange rate:")
    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "information",
                "risk_level": "low",
                "content": "The current exchange rate is: 1 USD = 135.50 JPY",
                "type": "exchange_rate",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    # Now query the exchange rates
    print("\nQuerying EUR exchange rate:")
    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "query",
                "risk_level": "low",
                "content": "What is the current USD to EUR exchange rate?",
                "type": "query",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    print("\nQuerying JPY exchange rate:")
    response = financial_agent.invoke(
        {
            "document_input": {
                "doc_type": "query",
                "risk_level": "low",
                "content": "What is the current USD to JPY exchange rate?",
                "type": "query",
                "metadata": {}
            }
        },
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()

    # Continue with transaction examples...
    # Example 1: First high-risk transaction
    test_transaction_high = {
        "doc_type": "transaction",
        "risk_level": "high",
        "content": "International wire transfer of $15,000",
        "amount": 15000,
        "type": "international",
        "metadata": {
            "transaction_id": "TX123"
        }
    }
    
    print("\nTesting first high-risk transaction:")
    response = financial_agent.invoke({"document_input": test_transaction_high}, config=config)
    for m in response["messages"]:
        m.pretty_print()

    # Example 2: Similar high-risk transaction (should reference the first one)
    test_transaction_similar = {
        "doc_type": "transaction",
        "risk_level": "high",
        "content": "International wire transfer of $16,000",
        "amount": 16000,
        "type": "international",
        "metadata": {
            "transaction_id": "TX125"
        }
    }
    
    print("\nTesting similar high-risk transaction:")
    response = financial_agent.invoke({"document_input": test_transaction_similar}, config=config)
    for m in response["messages"]:
        m.pretty_print()

    # Example 3: Low-risk transaction
    test_transaction_low = {
        "doc_type": "transaction",
        "risk_level": "low",
        "content": "Regular monthly salary deposit",
        "amount": 2500,
        "type": "wire",
        "metadata": {
            "transaction_id": "TX124"
        }
    }
    
    print("\nTesting low-risk transaction:")
    response = financial_agent.invoke({"document_input": test_transaction_low}, config=config)
    for m in response["messages"]:
        m.pretty_print()
        
    # Print all memories at the end
    print("\nFinal Memory State:")
    print_all_memories() 