# ai_ensemble_suite/config/templates.py

"""Prompt templates for different collaboration patterns."""

# Basic prompt templates
BASIC_TEMPLATES = {
    # General template for initial response
    "single_query": "You are an AI assistant. Answer the following question or respond to the instruction:\n\n{{ query }}",
    
    # Template for self-evaluation
    "self_evaluation": """You previously provided this response:

{{ response }}

Evaluate your confidence in this response on a scale from 1-10, where:
1 = Completely uncertain or guessing
10 = Absolutely certain based on verified facts

Provide ONLY a numeric rating, nothing else.""",
}

# Templates for debate-based collaboration
DEBATE_TEMPLATES = {
    # Template for initial response in a debate
    "debate_initial": """You are an AI assistant with expertise in providing balanced, thoughtful responses. 
Address the following query with a well-reasoned response:

QUERY: {{ query }}

Provide a comprehensive response that considers multiple perspectives.""",
    
    # Template for critique in a debate
    "debate_critique": """You are a thoughtful critic. Review the following response to the question and provide constructive criticism.

ORIGINAL QUESTION: {{ query }}

RESPONSE TO EVALUATE:
{{ response }}

Critically evaluate this response focusing on:
1. Factual accuracy - Are there any errors or misleading statements?
2. Comprehensiveness - Does it address all relevant aspects of the question?
3. Logical reasoning - Is the argument structure sound and coherent?
4. Fairness - Does it present a balanced view or show bias?
5. Clarity - Is the response clear and well-organized?

Provide specific, actionable feedback for improvement.""",
    
    # Template for a response to critique
    "debate_defense": """You are the original responder to a question that has received critique.

ORIGINAL QUESTION: {{ query }}

YOUR ORIGINAL RESPONSE:
{{ response }}

CRITIC'S FEEDBACK:
{{ critique }}

Respond to these criticisms by either:
1. Defending your original points with additional evidence and reasoning, or
2. Acknowledging valid criticisms and refining your position

Provide a thoughtful, balanced response to the critiques while maintaining intellectual integrity.""",
    
    # Template for synthesis after debate
    "debate_synthesis": """You are a neutral synthesizer reviewing a debate on the following question:

ORIGINAL QUESTION: {{ query }}

INITIAL RESPONSE:
{{ response }}

CRITIQUE:
{{ critique }}

DEFENSE:
{{ defense }}

Based on this exchange, provide a balanced synthesis that:
1. Identifies areas of agreement between the perspectives
2. Acknowledges legitimate differences
3. Presents the strongest version of the final answer that incorporates valid points from all sides
4. Notes any remaining uncertainties or areas where further information would be valuable

Your goal is to produce the most accurate and balanced perspective possible.""",
}

# Templates for role-based collaboration
ROLE_TEMPLATES = {
    # Template for researcher role
    "role_researcher": """You are a thorough researcher. For the following query, gather comprehensive, relevant facts. 
Focus on established knowledge, important context, and different perspectives.

QUERY: {{ query }}

Provide detailed information with a focus on accuracy and completeness.""",
    
    # Template for analyst role
    "role_analyst": """You are a critical analyst. Review the following research information and evaluate its quality,
identify potential flaws, and suggest areas needing further investigation.

ORIGINAL QUESTION: {{ query }}

RESEARCH INFORMATION:
{{ research }}

Analyze this information for:
1. Logical consistency
2. Potential biases
3. Evidence quality
4. Alternative interpretations
5. Knowledge gaps

Provide a structured analysis that identifies strengths and weaknesses.""",
    
    # Template for synthesizer role
    "role_synthesizer": """You are a knowledge synthesizer tasked with creating a coherent, comprehensive response
based on research and analysis.

ORIGINAL QUESTION: {{ query }}

RESEARCH INFORMATION:
{{ research }}

CRITICAL ANALYSIS:
{{ analysis }}

Create a synthesis that:
1. Integrates the factual information with analytical insights
2. Acknowledges uncertainties and limitations
3. Presents a coherent, balanced view of the topic
4. Organizes information in a logical, accessible way

Your goal is to produce the most accurate and useful response to the original question.""",
}

# Templates for hierarchical review
HIERARCHICAL_TEMPLATES = {
    # Template for initial draft
    "hierarchical_draft": """You are tasked with creating an initial draft response to the following query:

QUERY: {{ query }}

Provide a clear, straightforward response that addresses the core question.""",
    
    # Template for technical review
    "hierarchical_technical_review": """You are a technical specialist reviewing a draft response.

ORIGINAL QUERY: {{ query }}

DRAFT RESPONSE:
{{ draft }}

Review this response for technical accuracy. Identify any:
1. Technical errors or misunderstandings
2. Oversimplifications of complex concepts
3. Missing technical details that would improve the response

Provide specific corrections and suggestions for technical improvements.""",
    
    # Template for clarity review
    "hierarchical_clarity_review": """You are a clarity expert reviewing a response.

ORIGINAL QUERY: {{ query }}

CURRENT RESPONSE:
{{ response }}

Review this response for clarity and accessibility. Identify any:
1. Confusing explanations or jargon without proper explanation
2. Poorly structured information
3. Opportunities to improve readability

Suggest specific improvements to make this response clearer and more accessible.""",
    
    # Template for final refinement
    "hierarchical_final_refinement": """You are tasked with creating the final, polished response.

ORIGINAL QUERY: {{ query }}

CURRENT RESPONSE:
{{ response }}

TECHNICAL REVIEW:
{{ technical_review }}

CLARITY REVIEW:
{{ clarity_review }}

Using these reviews, create a final response that:
1. Incorporates the technical corrections and improvements
2. Implements the clarity suggestions
3. Provides the most accurate and accessible answer to the original query

Produce a complete, refined response for the user.""",
}

# Combine all template dictionaries
ALL_TEMPLATES = {
    **BASIC_TEMPLATES,
    **DEBATE_TEMPLATES,
    **ROLE_TEMPLATES,
    **HIERARCHICAL_TEMPLATES,
}
