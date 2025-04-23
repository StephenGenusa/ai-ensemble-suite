# AI Ensemble Framework: Complete Configuration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Model Configuration](#model-configuration)
3. [Templating Basics](#templating-basics)
4. [Advanced Template Construction](#advanced-template-construction)
5. [Collaboration Patterns](#collaboration-patterns)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Introduction

The AI Ensemble Framework allows you to orchestrate multiple AI models in collaborative workflows, leveraging each model's strengths while mitigating individual weaknesses. This guide covers how to configure models and templates to create powerful AI ensembles through YAML configuration files.

## Model Configuration

### Defining Models in YAML

Models form the foundation of the AI Ensemble Framework and must be specified in the `models` section of your YAML configuration:

```yaml
models:
  mistral:
    parameters:
      max_tokens: 4096
      n_ctx: 32768
      n_gpu_layers: -1
      temperature: 0.7
      top_p: 0.9
    path: C:\Users\Username\models\Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
    role: primary
  deepseek:
    parameters:
      max_tokens: 4096
      n_ctx: 32768
      n_gpu_layers: -1
      temperature: 0.75
      top_p: 0.9
    path: C:\Users\Username\models\DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
    role: critic
```

### VRAM Considerations and Performance

**Important**: The number of models you can load simultaneously is primarily dictated by your available VRAM (GPU memory). Key considerations:

- **VRAM Constraints**: Each model requires VRAM proportional to its size and quantization level, and other parameters such as the size of the context window you allocate
- **Multiple Models**: Loading multiple models simultaneously requires sufficient VRAM
- **CPU Fallback**: While llama.cpp supports loading models to system RAM using `n_gpu_layers: 0`, inference on CPU is extremely slow and not recommended for practical use
- **Partial GPU Offloading**: Setting `n_gpu_layers` to a value between 1 and the total number of layers allows partial GPU acceleration
- **Auto Configuration**: Setting `n_gpu_layers: -1` attempts to automatically use as much GPU as available

For optimal performance with multiple models:
- Use quantized models (Q4_K_M, Q5_K_M, Q6_K) to reduce VRAM requirements
- Consider smaller model variants (7B instead of 13B or larger)
- Use `n_gpu_layers` strategically to balance models across available VRAM

### Key Model Configuration Parameters

Each model requires specific configuration parameters:

#### Required Parameters
- **`path`**: Path to the model file (GGUF format for local models)
  - Must be a valid file path accessible to the application
  - Can be relative or absolute path
  
- **`role`**: Functional role within the ensemble (string value)
  - Common roles: "primary", "critic", "reasoning", "evaluator"
  - User-definable based on your ensemble design

#### Optional Parameters
- **`parameters`**: Inference settings for the model
  - `temperature`: Controls randomness (0.0-2.0)
    - Lower values (0.1-0.4): More deterministic, factual responses
    - Higher values (0.7-1.0): More creative, diverse responses
  
  - `top_p`: Nucleus sampling parameter (0.0-1.0)
    - Lower values: More focused sampling from most likely tokens
    - Higher values: Broader vocabulary diversity
  
  - `top_k`: Limits vocabulary to top K tokens (integer)
    - Lower values: More focused responses
    - Higher values (or 0): More diverse vocabulary
  
  - `max_tokens`: Maximum response length (integer)
    - Limits how long model responses can be
    - Should be balanced against context length
  
  - `n_ctx`: Context window size (integer)
    - Maximum combined length of prompt and response
    - Higher values need more VRAM
  
  - `n_gpu_layers`: Number of layers to offload to GPU (integer)
    - `-1`: Auto-determine (use as much GPU as possible)
    - `0`: CPU only (very slow)
    - Partial values: Balance between GPU and CPU
  
  - `repeat_penalty`: Penalty for repetition (float, usually ≥1.0)
    - Higher values discourage repetitive text

### Model-Template Relationship

Models and templates are connected in the `collaboration` section through phases. Each phase defines:

1. Which **model(s)** to use
2. Which **template** to apply
3. The **phase type** that affects template behavior

```yaml
collaboration:
  phases:
  - models:
    - mistral
    name: initial_response
    prompt_template: debate_initial
    type: async_thinking
```

In this example:
- The `mistral` model processes the `debate_initial` template
- The phase type `async_thinking` indicates this is an independent processing step

## Templating Basics

### What Are Templates?

In the AI Ensemble Framework, templates are pre-defined text patterns that:
- Contain instructions for AI models
- Include placeholders for dynamic content
- Structure the flow of information between models
- Define the format of responses

Templates use Jinja2, a powerful templating engine that allows for dynamic content insertion and basic logic.

### Jinja2 Basics

The framework uses Jinja2 templating syntax. Here are the key elements:

#### Variables
Variables are placeholders that get replaced with actual values when the template is rendered:
```
{{ variable_name }}
```

Common variables you'll see in templates include:
- `{{ query }}` - The user's original question or instruction
- `{{ response }}` - A model's response to a query
- `{{ critique }}` - Feedback from a model about another response

#### Control Structures
Jinja2 allows for conditional rendering and loops:

**Conditionals:**
```
{% if condition %}
  Content to include if condition is true
{% else %}
  Alternative content
{% endif %}
```

**Loops:**
```
{% for item in items %}
  {{ item }}
{% endfor %}
```

#### Comments
```
{# This is a comment that won't appear in the rendered template #}
```

### Template Structure

A basic template in the AI Ensemble Framework typically includes:

1. **System Instructions**: Define the AI model's role and behavior
2. **Context Information**: Provide necessary background or results from previous phases
3. **Query**: Present the user's original question
4. **Output Format Instructions**: Specify how the response should be structured

### Example: Basic Template

Here's a simple template from `basic_config.yaml`:

```yaml
single_query: 'You are a helpful AI assistant. Please answer the following question:


{{ query }}'
```

This template simply:
1. Instructs the model to act as a helpful assistant
2. Provides the user query through the `{{ query }}` variable

### Referencing Templates in Configuration

In YAML configuration files, templates are:

1. **Defined** in the `templates` section
2. **Referenced** in collaboration phases via `prompt_template`

Example from `basic_config.yaml`:

```yaml
collaboration:
  phases:
  - name: initial_response
    type: async_thinking
    models:
    - mistral
    prompt_template: single_query  # Reference to the template
```

### Model-Specific Templating

Templates can be designed with specific models in mind:

#### 1. Adapting to Model Capabilities

Different models have different strengths, so templates should be tailored accordingly:

```yaml
templates:
  # For models good at reasoning
  reasoning_template: 'You excel at step-by-step reasoning. Break down this problem:

  {{ query }}
  
  Think through each step carefully, showing your work.'
  
  # For models good at summarization
  summarization_template: 'You are excellent at concise summarization. Summarize this:
  
  {{ text_to_summarize }}
  
  Create a clear, concise summary capturing the key points.'
```

#### 2. Model-Specific Instructions

Templates can include special instructions for specific models:

```yaml
special_instruction_template: |
  {% if model_name == "mistral" %}
  You are Mistral, optimized for logical reasoning. Use your logical abilities to solve this problem.
  {% elif model_name == "deepseek" %}
  You are DeepSeek, with strong critical analysis skills. Apply your critical thinking to evaluate this.
  {% else %}
  You are an AI assistant. Please help with the following:
  {% endif %}
  
  {{ query }}
```

## Advanced Template Construction

### Template System Architecture

The template system consists of:

1. **Template Definitions**: Either hard-coded in `templates.py` or user-defined in YAML
2. **Template Manager**: Handles retrieval and rendering of templates
3. **Template Engine**: Renders Jinja2 templates with the appropriate context
4. **Validation Rules**: Ensure templates meet system requirements

### Hard-Coded vs. User-Defined Templates

#### Hard-Coded Templates in `templates.py`

The framework includes several built-in templates in the `templates.py` file:

1. **Basic Templates**:
   - `single_query`: Simple query template
   - `self_evaluation`: For confidence assessment

2. **Debate Templates**:
   - `debate_initial`: Initial response
   - `debate_critique`: Critical analysis
   - `debate_defense`: Response to critique
   - `debate_synthesis`: Integration of perspectives

3. **Role-Based Templates**:
   - `role_researcher`: For fact gathering
   - `role_analyst`: For critical analysis
   - `role_synthesizer`: For knowledge integration

4. **Hierarchical Templates**:
   - `hierarchical_draft`: Initial draft
   - `hierarchical_technical_review`: Technical review
   - `hierarchical_clarity_review`: Clarity review
   - `hierarchical_final_refinement`: Final polish

#### User-Defined Templates in YAML

Users can define custom templates in the `templates` section of a YAML configuration file. These **override** any hard-coded templates with the same name.

Example from `advanced_prompting_example.yaml`:

```yaml
templates:
  comparative_analysis: 'You''ve provided two responses to the query:


    {{ query }}


    {% if basic_response %}

    Response 1 (Standard approach):

    {{ basic_response }}

    {% else %}

    Response 1: [No standard approach response available]

    {% endif %}


    {% if cot_response %}

    Response 2 (Step-by-step reasoning):

    {{ cot_response }}

    {% else %}

    Response 2: [No step-by-step reasoning response available]

    {% endif %}


    Now, compare these approaches and create a final response that combines the strengths
    of both. Focus on accuracy, clarity, and depth of explanation.'
```

### Required Template Elements

While the content of templates is largely up to the user, certain key elements are expected based on the validation logic in `schema.py`:

1. **Templates must be non-empty strings**
2. **Templates referenced in the configuration must exist**
3. **Templates used in specific phases should handle the expected input variables**

### Context and Variable Mapping

Understanding which variables are available in each template is crucial:

#### Core Variables (Always Available)
- `query`: The original user query

#### Phase-Specific Variables
Variables available depend on the phase type and configuration:

1. **`async_thinking` phase**: Limited to core variables
2. **`integration` phase**: Receives outputs from previous phases via `input_from`
3. **`structured_debate` phase**: Gets debate-specific variables 
4. **`bagging` phase**: Receives aggregation results

Example from `debate_config.yaml`:
```yaml
- input_from:
  - initial_response
  - critique
  models:
  - deepseek
  name: defense
  prompt_template: debate_defense
  subtype: synthesis
  type: structured_debate
```

In this example, the `debate_defense` template would have access to:
- `query`: The original question
- `initial_response`: Output from the initial response phase
- `critique`: Output from the critique phase

### Advanced Jinja2 Features

The framework's template engine supports several advanced Jinja2 features:

#### Custom Filters
Custom filters provide additional functionality:

1. **`substring`**: Extract part of a string
   ```
   {{ some_long_text|substring(0, 1000) }}
   ```

2. **`truncate`**: Limit string length
   ```
   {{ response|truncate(500) }}
   ```

#### Conditional Logic for Robust Templates

Templates should handle cases where expected inputs might be missing:

```
{% if variable_name %}
  {{ variable_name }}
{% else %}
  Default text or error message
{% endif %}
```

Example from `advanced_prompting_example.yaml`:
```
{% if basic_response %}
Response 1 (Standard approach):
{{ basic_response }}
{% else %}
Response 1: [No standard approach response available]
{% endif %}
```

## Collaboration Patterns

The AI Ensemble Framework supports several collaboration patterns, each requiring specific template structures.

### Sequential Refinement

In sequential refinement, templates build on previous outputs, with each phase adding refinement. This is seen in `advanced_prompting_example.yaml`:

```yaml
collaboration:
  mode: advanced_prompting
  phases:
  - models:
    - mistral
    name: basic_response
    prompt_template: expert_prompt
    type: async_thinking
  - models:
    - deepseek
    name: cot_response
    prompt_template: cot_prompt
    type: async_thinking
  - input_from:
    - basic_response
    - cot_response
    models:
    - mistral
    name: comparative_analysis
    prompt_template: comparative_analysis
    type: integration
```

### Structured Debate

The debate pattern involves templates with specific roles in a discussion process:

```yaml
collaboration:
  mode: structured_debate
  phases:
  - models:
    - mistral
    name: initial_response
    prompt_template: debate_initial
    type: async_thinking
  - input_from: initial_response
    models:
    - deepseek
    name: critique
    prompt_template: debate_critique
    subtype: critique
    type: structured_debate
  - input_from:
    - initial_response
    - critique
    models:
    - deepseek
    name: defense
    prompt_template: debate_defense
    subtype: synthesis
    type: structured_debate
```

### Bagging (Ensemble Method)

Bagging generates multiple variations and performs meta-analysis:

```yaml
collaboration:
  mode: custom
  phases:
  - aggregation_method: voting
    models:
    - mistral
    - deepseek
    - llama
    name: bagged_responses
    num_variations: 4
    prompt_template: bagging_prompt
    sample_ratio: 0.9
    type: bagging
    variation_strategy: instruction_variation
  - input_from:
    - bagged_responses
    models:
    - mistral
    name: meta_analysis
    prompt_template: meta_analysis_prompt
    type: integration
```

The meta-analysis template processes outputs from bagging:

```yaml
templates:
  meta_analysis_prompt: 'You are a meta-analyzer tasked with synthesizing multiple
    AI responses to the same query.

    ...

    {% for output in bagged_responses.outputs %}

    --- MODEL RESPONSE {{ loop.index }} ---

    {{ output|substring(0, 1200) }}...


    {% endfor %}

    ...'
```

### How Model Parameters Affect Templating

The model parameters can influence how templates should be designed:

#### Temperature and Response Determinism

- **Low temperature** (0.1-0.4): More deterministic responses, better for factual outputs
- **Medium temperature** (0.5-0.8): Balanced creativity and coherence
- **High temperature** (0.9-1.5): More creative/diverse outputs

Templates should be designed with the temperature setting in mind:

```yaml
# For low temperature/deterministic model
structured_template: 'Provide a factual, point-by-point analysis of:

{{ query }}

Format your response in numbered points, focusing on verified facts only.'

# For high temperature/creative model
creative_template: 'Generate diverse, creative solutions to:

{{ query }}

Think outside the box and provide multiple unique approaches.'
```

#### Context Window and Template Size

The `n_ctx` parameter defines how much context a model can process. Templates must be designed with this limitation in mind:

```yaml
# For models with small context windows
compact_template: 'Answer concisely:
{{ query }}'

# For models with large context windows
detailed_template: 'Provide a comprehensive analysis of the following question.
Include historical context, current perspectives, and future implications:

{{ query }}

In your response, consider multiple viewpoints and address potential limitations or counterarguments.'
```

## Troubleshooting

### Common Model-Template Integration Issues

1. **Model Compatibility Issues**: Ensure templates match the capabilities of the assigned models
   ```
   # May not work well with smaller models
   complex_reasoning_template: 'Perform a detailed analysis with 20 logical steps...'
   ```

2. **Parameter Mismatch**: Adjust templates to match model parameters
   ```
   # Too complex for high temperature setting
   factual_template: 'Provide precise facts about {{ query }} without any speculation.'
   ```

3. **Context Overflow**: Templates + inputs exceeding model's context window
   ```
   # Solution: Use substring filter
   long_content_template: '{{ previous_content|substring(0, 2000) }}'
   ```

4. **Model Role Mismatch**: Templates not aligned with the model's designated role
   ```
   # Misalignment example: critic model with primary template
   mistaken_config:
     models:
       critic_model:
         role: critic
     phases:
     - models:
       - critic_model
       prompt_template: primary_response_template  # Should use a critique template
   ```

### Other Common Issues

1. **Missing Variables**: Check that your template only references variables that will be available
2. **Template Not Found**: Ensure the template is defined in `templates` section or exists in `templates.py`
3. **Incorrect Phase Order**: Phase outputs are only available to later phases via `input_from`
4. **Rendering Errors**: Check Jinja2 syntax, especially unmatched brackets or quotes
5. **VRAM Overflow**: Trying to load too many models simultaneously

## Best Practices

### Model Configuration Best Practices

1. **Quantization Level Selection**
   - Higher quantization (Q8_0) = Better quality but more VRAM
   - Lower quantization (Q4_K_M) = Lower VRAM but slightly reduced quality
   - Recommendation: Start with Q6_K for balance or Q4_K_M for VRAM constraints

2. **VRAM Management**
   - Monitor VRAM usage during initialization
   - Use `n_gpu_layers` to fine-tune GPU utilization
   - Consider model size when designing multi-model ensembles

3. **Model Role Specialization**
   - Assign appropriate roles based on model strengths
   - Use larger/more capable models for complex reasoning tasks
   - Use specialized models for specific functions (critique, summarization)

### Template Design Best Practices

1. **Clear Role Definition**
   Start templates with clear instructions about the model's role:
   ```
   You are an expert in [domain]. Your task is to [specific action].
   ```

2. **Structured Input Processing**
   For templates that process outputs from previous phases:
   ```
   I'll provide you with [type of content] and your task is to [action].

   ORIGINAL QUERY:
   {{ query }}

   PREVIOUS RESPONSE:
   {{ previous_phase_output }}
   ```

3. **Format Specification**
   Be explicit about the expected output format:
   ```
   Format your response as:
   1. [Section 1]
   2. [Section 2]
   ...
   ```

4. **Robust Error Handling**
   Include fallbacks for missing content:
   ```
   {% if variable %}
     {{ variable }}
   {% else %}
     [No content available for this section]
   {% endif %}
   ```

5. **Match Template Complexity to Model Capabilities**
   - Larger models → More complex reasoning tasks
   - Smaller models → More focused, specific tasks

6. **Adjust Instruction Style to Model Training**
   - Some models respond better to direct instructions
   - Others perform better with role-playing prompts

7. **Consider Parameter Settings When Designing Templates**
   - Higher temperature → More open-ended templates
   - Lower temperature → More structured templates

### Complete Example: Building a Debate Configuration

Let's analyze the complete structure of the debate configuration from `debate_config.yaml`:

1. **Model Configuration**:
   ```yaml
   models:
     deepseek:
       parameters:
         max_tokens: 4096
         n_ctx: 32768
         n_gpu_layers: -1
         temperature: 0.75
         top_p: 0.9
       path: C:\Users\Stephen\PycharmProjects\ensemble-ai\models\DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
       role: critic
     mistral:
       parameters:
         max_tokens: 4096
         n_ctx: 32768
         n_gpu_layers: -1
         temperature: 0.75
         top_p: 0.9
       path: C:\Users\Stephen\PycharmProjects\ensemble-ai\models\Mistral-7B-Instruct-v0.3-Q6_K.gguf
       role: critic
   ```

2. **Collaboration Structure**:
   ```yaml
   collaboration:
     mode: structured_debate
     phases:
     - models:
       - mistral
       name: initial_response
       prompt_template: debate_initial
       type: async_thinking
     - input_from: initial_response
       models:
       - deepseek
       name: critique
       prompt_template: debate_critique
       subtype: critique
       type: structured_debate
     # Additional phases omitted for brevity...
   ```

3. **Templates**:
   ```yaml
   templates:
     debate_initial: "You are an AI assistant with expertise in providing balanced, thoughtful responses. 
     Address the following query with a well-reasoned response:

     QUERY: {{ query }}

     Provide a comprehensive but concise response that considers multiple perspectives."
     
     debate_critique: 'You are a thoughtful critic with expertise in critical analysis
       and reasoning.

       Review the following response to this question:

       ORIGINAL QUESTION: {{ query }}

       RESPONSE TO EVALUATE:
       {{ initial_response }}

       Critically evaluate this response by reasoning through:
       1. Factual accuracy - Are there any errors or misleading statements?
       2. Comprehensiveness - Does it address all relevant aspects of the question?
       3. Logical reasoning - Is the argument structure sound and coherent?
       4. Fairness - Does it present a balanced view or show bias?
       5. Clarity - Is the response clear and well-organized?

       Provide concise, actionable feedback for improvement.'
     
     # Additional templates omitted for brevity...
   ```

4. **Aggregation Configuration**:
   ```yaml
   aggregation:
     final_phase: improved_response
     strategy: sequential_refinement
   ```

### Creating Custom Collaboration Patterns

To create your own collaboration pattern:

1. **Define models** in the `models` section with appropriate paths and parameters
2. **Define phases** in the `collaboration` section
3. **Create templates** in the `templates` section
4. **Connect phases** using `input_from` references
5. **Select appropriate models** for each phase
6. **Configure aggregation** to determine the final output

### Conclusion

The AI Ensemble Framework provides a powerful system for orchestrating multiple AI models in collaborative workflows. By understanding how to configure models and design effective templates, you can create sophisticated ensemble systems that leverage the strengths of different models while mitigating their individual weaknesses.

The key to success is carefully matching templates to models and their parameters, designing clear communication flows between phases, and ensuring your hardware can support your chosen configuration. With these principles in mind, you can build ensemble AI systems that deliver superior results across a wide range of tasks.