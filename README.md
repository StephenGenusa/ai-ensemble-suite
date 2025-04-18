# AI Ensemble Suite

> **Alpha Release (v0.5)**: This library is in active development. While already functional and useful for real-world applications, some APIs may change, and additional cleanup and feature development are in progress. Feedback and contributions are welcome!

[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/StephenGenusa/ai-ensemble-suite)

Python framework for multiple GGUF language models to collaborate on tasks using structured communication patterns, aggregating their outputs into coherent responses. While I chose to support GGUF to begin with, I plan to add OpenAI compatible server support allowing local LLM server and Internet APIs to be called.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Ensembles Explained](#ai-ensembles-explained)
- [Why Ensembles Work](#why-ensembles-work)
- [Why This Library?](#why-this-library)
- [Examples](#examples)
- [Decision Guide: Choosing Collaboration & Aggregation Patterns](#decision-guide-choosing-collaboration--aggregation-patterns)
- [Collaboration Phases](#collaboration-phases)
- [Aggregation Strategies](#aggregation-strategies)
- [API Reference](#api-reference)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [Running Tests](#running-tests)
- [Roadmap](#roadmap)
- [License](#license)
- [Special Thanks](#special-thanks)

## Installation

### Prerequisites

- Python >= 3.10
- Sufficient RAM for running multiple language models

<!-- 
### Install via pip

```bash
pip install ai-ensemble-suite
```

### Installation with GPU acceleration

For NVIDIA GPU acceleration:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
pip install ai-ensemble-suite
```

For Apple Metal (macOS):

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
pip install ai-ensemble-suite
```
-->

### Development Installation

To install the package with development dependencies:

```bash
git clone https://github.com/StephenGenusa/ai-ensemble-suite.git
cd ai-ensemble-suite
pip install -e ".[dev]"
```

[Back to top](#ai-ensemble-suite)

## Quick Start

You will want to determine which GGUF models you wish to work with. Download them and place in the root folder of the project under a /models directory... or change the YAML to point to the location of your models.

Here's a minimal example to get started with AI Ensemble Suite:

```python
from ai_ensemble_suite import Ensemble

# Initialize the ensemble with a configuration
ensemble = Ensemble(config_path="path/to/config.yaml")

# Initialize models (loads them into memory)
await ensemble.initialize()

# Ask a question
try:
    response = await ensemble.ask("What are the key considerations for sustainable urban planning?")
    print(response)
finally:
    # Release resources when done
    await ensemble.shutdown()

# Alternatively, use with async context manager
async with Ensemble(config_path="path/to/config.yaml") as ensemble:
    response = await ensemble.ask("What are the key considerations for sustainable urban planning?")
    print(response)
```

[Back to top](#ai-ensemble-suite)

## Key Features

- ✅ **Multiple Collaboration Phases**: Choose from over a dozen different collaboration methods to design your ensemble approach
- ✅ **Specialized Model Roles**: Assign different roles (critic, synthesizer, researcher, etc.) to models, allowing them to specialize in different aspects of task completion
- ✅ **Flexible Aggregation Strategies**: Use different methods for combining model outputs into coherent, high-quality responses
- ✅ **Advanced Tracing**: Get detailed traces of model interactions and decision-making processes
- ✅ **Extensible Design**: Easily add custom collaboration phases or aggregation strategies
- ✅ **Built for GGUF Models**: Optimized for running multiple smaller GGUF language models locally
- ✅ **Confidence Estimation**: Token probability analysis and self-evaluation capabilities
- ✅ **Concurrent Model Management**: Efficient loading and execution of multiple models
- ✅ **Async-First Design**: Native async/await support with context manager

[Back to top](#ai-ensemble-suite)


## AI Ensembles Explained

Think of AI ensembles like getting advice from a group of experts instead of just one person. Just as you might ask several friends for opinions before making an important decision, AI ensembles combine the strengths of multiple AI systems or approaches to produce better results than any single AI could achieve alone. Instead of relying on one AI that might have blind spots or make certain types of mistakes, ensembles bring together diverse AI perspectives that can check each other's work, complement each other's strengths, and collectively arrive at more reliable answers. It's similar to how a team of doctors with different specialties might collaborate on a difficult medical case—the combined expertise leads to better outcomes than what any individual doctor could provide on their own.

**Brief Explanations of AI Ensemble Techniques**

* **AsyncThinking**: Like a rapid brainstorming session where multiple people jot down ideas independently before sharing, this technique generates diverse initial thoughts quickly without influence from other perspectives.

* **StructuredCritique**: Similar to having your work reviewed by a tough but fair editor, this approach systematically evaluates ideas, identifies logical flaws, and improves the rigor of thinking.

* **SynthesisOriented**: Acts like a skilled mediator who finds common ground between opposing viewpoints, integrating different perspectives into a balanced, comprehensive analysis.

* **RoleBasedDebate**: Resembles a panel discussion with experts from different fields, each contributing specialized knowledge to address complex topics from multiple angles.

* **HierarchicalReview**: Works like a multi-stage editing process, where content is refined layer by layer, with each review focusing on different aspects for progressive improvement.

* **CompetitiveEvaluation**: Functions like a contest where multiple solutions compete, and the strongest approach wins based on objective criteria.

* **PerspectiveRotation**: Similar to walking around a sculpture to view it from all sides, this technique examines issues from different stakeholder perspectives, ethical frameworks, or creative angles.

* **ChainOfThoughtBranching**: Like mapping out a complex maze with multiple possible paths, this method explores different reasoning routes for problems with multiple decision points.

* **AdversarialImprovement**: Acts as a stress-test or devil's advocate, actively looking for weaknesses in a solution to strengthen it against potential problems.

* **RoleBasedWorkflow**: Operates like a production line with specialized stations, creating a structured process where different roles handle specific aspects of a multi-stage analysis.

* **Bagging**: Works like taking the average of multiple poll results to get a more stable prediction, reducing the impact of outliers or unusual patterns.

* **UncertaintyBasedCollaboration**: Similar to how a group might work together on a puzzle with missing pieces, this approach handles ambiguous questions by combining different levels of confidence.

* **StackedGeneralization**: Functions like a team of specialists with a coordinator, where outputs from different AI models are combined to leverage their unique strengths and minimize weaknesses.

[Back to top](#ai-ensemble-suite)

## Why Ensembles Work

Ensemble methods have a strong theoretical and empirical foundation in machine learning, and they're particularly effective with language models for several reasons:

1. **Diverse Knowledge and Perspectives**: Different language models, even when trained on similar data, develop slightly different internal representations and "expertise areas." By combining multiple models, you access a broader knowledge base than any single model contains.

2. **Error Reduction Through Aggregation**: Models tend to make different errors. When their outputs are combined intelligently, errors from one model can be corrected by others, leading to more accurate results.

3. **Specialization Through Roles**: When models adopt specialized roles (like critic, researcher, or synthesizer), they can focus on specific aspects of a task. This division of cognitive labor mirrors effective human teams and leads to more thorough analysis.

4. **Iterative Refinement**: Multi-step collaboration allows initial ideas to be critiqued, refined, and expanded. This resembles human drafting and editing processes, typically producing higher quality results than single-pass generation.

5. **Confidence Calibration**: Ensemble techniques help identify areas of uncertainty or disagreement between models, leading to better-calibrated confidence in the final output.

Research consistently shows that properly designed ensembles outperform even the strongest individual models, often by significant margins. AI Ensemble Suite provides the infrastructure to easily tap into these powerful techniques.

[Back to top](#ai-ensemble-suite)

## Why This Library?

AI Ensemble Suite was created to address two key challenges:

1. **The Need for Human-Friendly Ensemble AI**: After extensive searching for a comprehensive yet easy-to-use library for ensemble AI work that followed a "for humans" philosophy, nothing quite fit the bill. This library makes it easy to harness multiple smaller language models on local machines to produce enhanced AI responses.

2. **Structured Collaboration Patterns**: Rather than just averaging model outputs, AI Ensemble Suite implements sophisticated collaboration patterns where models can critique, refine, and extend each other's work - resulting in higher quality responses that benefit from diverse model strengths.

The framework is designed with simplicity in mind for common use cases while providing comprehensive customization for advanced users.

Additionally, this project served as a meta-challenge to build a medium sized Python library using AI assistance despite context window limitations, developing techniques to work around these constraints.

[Back to top](#ai-ensemble-suite)

## Examples

The library includes several example scripts demonstrating different ensemble techniques:

- **Basic Usage**: Non-ensembled simple usage with default configurations
- **Structured Debate**: Models present opposing viewpoints to refine conclusion
- **Expert Committee**: Specialized models contribute domain expertise
- **Hierarchical Review**: Progressive refinement through layers of specialized models
- **Competitive Evaluation**: Multiple solutions generated and evaluated against criteria
- **Perspective Rotation**: Problem analyzed through different framing lenses
- **Chain-of-Thought Branching**: Reasoning paths that branch and reconverge
- **Adversarial Improvement**: One model finds flaws in another's reasoning
- **Role-based Workflow**: Models adopt complementary roles in a structured process
- **Bagging**: Combines models to reduce prediction variance

> **Note on Examples**: Some example files require additional libraries not included in requirements.txt. Several examples also generate graphic charts to disk, which were implemented to help visualize how the ensemble builds the final result. These visualization features are optional but helpful for testing and understanding the process.

[Back to top](#ai-ensemble-suite)

## Decision Guide: Choosing Collaboration & Aggregation Patterns

### Collaboration Patterns

| When you need... | Use this collaboration pattern | Best for |
|------------------|-------------------------------|----------|
| Quick independent analyses | **AsyncThinking** | Simple questions, brainstorming, diverse initial ideas |
| Critical evaluation of ideas | **StructuredCritique** | Evaluating arguments, finding flaws, improving rigor |
| Balanced perspectives | **SynthesisOriented** | Finding common ground, integrating viewpoints, balanced analysis |
| Multiple specialist perspectives | **RoleBasedDebate** | Complex topics requiring multiple forms of expertise |
| Progressive improvement | **HierarchicalReview** | Content requiring layer-by-layer refinement or fact-checking |
| Competition between solutions | **CompetitiveEvaluation** | Generating multiple solutions and selecting the best one |
| Examining from different angles | **PerspectiveRotation** | Ethical analysis, stakeholder considerations, creative ideation |
| Complex reasoning paths | **ChainOfThoughtBranching** | Mathematical problems, logic puzzles, decision trees |
| Stress-testing solutions | **AdversarialImprovement** | Finding edge cases, improving robustness, anticipating objections |
| Structured workflow process | **RoleBasedWorkflow** | Research projects, content creation, multi-stage analysis |
| Stabilizing volatile outputs | **Bagging** | Reducing variance, improving prediction stability |
| Handling uncertainty | **UncertaintyBasedCollaboration** | Questions with ambiguity, calibrating confidence |
| Model stacking | **StackedGeneralization** | Leveraging strengths of different model types, boosting performance |

### Aggregation Strategies

| When you need... | Use this aggregation strategy | Best for |
|------------------|------------------------------|----------|
| To prioritize some models over others | **WeightedVoting** | When certain models perform better for specific tasks |
| To use the final result of a sequence | **SequentialRefinement** | When using phases that progressively refine content |
| To choose the most confident output | **ConfidenceBased** | When models have reliable confidence estimation |
| To evaluate along multiple criteria | **MultidimensionalVoting** | Complex evaluation requiring different quality dimensions |
| To blend multiple perspectives | **EnsembleFusion** | Creating a coherent synthesis from diverse inputs |
| Dynamic strategy selection | **AdaptiveSelection** | When different queries benefit from different aggregation approaches |

[Back to top](#ai-ensemble-suite)

## Collaboration Phases

AI Ensemble Suite implements various collaboration phases that can be combined or used individually:

### Core Phases

- ✅ **AsyncThinking**: Models work independently on a problem before combining insights
- ✅ **Integration/Refinement**: Models refine responses based on feedback and insights
- ✅ **ExpertCommittee**: Final processing/structuring of model outputs before aggregation

### Structured Debate Patterns

- ✅ **StructuredCritique**: Models evaluate others' responses using structured formats
- ✅ **SynthesisOriented**: Models focus on finding common ground and integrating perspectives
- ✅ **RoleBasedDebate**: Models interact according to assigned specialized roles

### Advanced Collaboration Methods

- ✅ **HierarchicalReview**: Content is progressively reviewed by models in a hierarchical structure
- ✅ **CompetitiveEvaluation**: Models are pitted against each other in a competition
- ✅ **PerspectiveRotation**: Models iterate on a problem by assuming different perspectives
- ✅ **ChainOfThoughtBranching**: Models trace through multiple reasoning paths
- ✅ **AdversarialImprovement**: Models improve a solution by seeking its weaknesses
- ✅ **RoleBasedWorkflow**: Models function in specialized roles like researcher, analyst, and writer
- ✅ **Bagging**: Models process different variations of the same input
- ✅ **UncertaintyBasedCollaboration**: Uncertainty measurements guide model interactions
- ✅ **StackedGeneralization**: Base models process input, then a meta-model combines their outputs

[Back to top](#ai-ensemble-suite)

## Aggregation Strategies

The library provides several strategies for aggregating the outputs from multiple models:

- ✅ **WeightedVoting**: Models are assigned different weights based on performance or expertise
- ✅ **SequentialRefinement**: Assumes phases run in a sequence where later phases refine earlier ones
- ✅ **ConfidenceBased**: Selects output with the highest confidence score
- ✅ **MultidimensionalVoting**: Evaluates outputs along multiple dimensions
- ✅ **EnsembleFusion**: Uses a model to synthesize multiple outputs into one
- ✅ **AdaptiveSelection**: Dynamically selects and executes another aggregation strategy

[Back to top](#ai-ensemble-suite)

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│                                             │
│            Ensemble Class                   │
│     (Main user-facing interface)            │
│                                             │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────┐ │ ┌─────────────────────┐
│                 │ │ │                     │
│  ConfigManager  │◄┼─┤   ModelManager      │
│  (YAML configs) │ │ │   (GGUF Models)     │
│                 │ │ │                     │
└─────────────────┘ │ └──────────┬──────────┘
                    │            │
                    │            ▼
┌───────────────────┴────────────────────────┐
│                                            │
│          Collaboration Pipeline            │
│                                            │
│  ┌──────────────┐     ┌──────────────┐     │
│  │ Phase 1      │     │ Phase 2      │     │
│  │ AsyncThinking│────►│ Debate...    │────►│...
│  │              │     │              │     │
│  └──────────────┘     └──────────────┘     │
│                                            │
└────────────────────────┬─────────────────┬─┘
                         │                 │
                         ▼                 │
┌───────────────────────────────────────┐  │
│                                       │  │
│         Aggregation Strategy          │  │
│                                       │  │
│  ┌─────────────────────────────────┐  │  │
│  │ WeightedVoting, EnsembleFusion, │  │  │
│  │ SequentialRefinement, etc.      │  │  │
│  └─────────────────────────────────┘  │  │
│                                       │  │
└───────────────────┬───────────────────┘  │
                    │                      │
                    ▼                      │
┌───────────────────────────────────────┐  │
│                                       │  │
│           Final Response              │  │
│                                       │  │
└───────────────────────────────────────┘  │
                                           │
┌───────────────────────────────────────┐  │
│                                       │  │
│          Tracing System               │◄─┘
│    (Records all collaboration steps)  │
│                                       │
└───────────────────────────────────────┘
```

[Back to top](#ai-ensemble-suite)

## API Reference

### Core Classes

#### Ensemble

```python
class Ensemble:
    """Coordinates the collaboration of multiple AI models for complex tasks."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Ensemble orchestration layer."""
        
    async def initialize(self) -> None:
        """Load models and prepare the ensemble for processing queries."""
        
    async def shutdown(self) -> None:
        """Release resources used by the ensemble."""
        
    async def ask(self, query: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """Process a query through the configured collaboration and aggregation pipeline."""
        
    def configure(self, config_dict: Dict[str, Any]) -> None:
        """Update the ensemble's configuration dynamically."""
```

#### ModelManager

```python
class ModelManager:
    """Manages the loading, execution, and lifecycle of GGUF models."""
    
    def __init__(self, config_manager: ConfigProvider, max_workers: Optional[int] = None) -> None:
        """Initialize the ModelManager."""
        
    async def initialize(self) -> None:
        """Initialize the ModelManager: Instantiates models and loads them asynchronously."""
        
    async def shutdown(self) -> None:
        """Shutdown the ModelManager: Unloads models and shuts down the executor."""
        
    async def run_inference(self, model_id: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Run inference on a specific model using its generate method via thread pool."""
```

#### BaseCollaborationPhase

```python
class BaseCollaborationPhase(ABC):
    """Abstract base class for collaboration phases."""
    
    def __init__(self, model_manager: "ModelManager", config_manager: "ConfigManager", phase_name: str) -> None:
        """Initialize the collaboration phase."""
        
    @abstractmethod
    async def execute(self, query: str, context: Dict[str, Any], trace_collector: Optional[TraceCollector] = None) -> Dict[str, Any]:
        """Execute the collaboration phase."""
```

#### BaseAggregator

```python
class BaseAggregator(ABC):
    """Abstract base class for aggregation strategies."""
    
    def __init__(self, config_manager: "ConfigManager", strategy_name: str, model_manager: Optional["ModelManager"] = None, strategy_config_override: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the aggregator."""
        
    @abstractmethod
    async def aggregate(self, outputs: Dict[str, Dict[str, Any]], context: Dict[str, Any], trace_collector: Optional[TraceCollector] = None) -> Dict[str, Any]:
        """Aggregate the outputs from collaboration phases."""
```

[Back to top](#ai-ensemble-suite)

## Requirements

- Python >= 3.10
- llama-cpp-python >= 0.2.0
- pyyaml >= 6.0
- asyncio >= 3.4.3

### Optional Dependencies

The library has optional dependencies that can be installed with:

```bash
pip install "ai-ensemble-suite[rich]"  # For enhanced console output
pip install "ai-ensemble-suite[dev]"   # For development tools
```

[Back to top](#ai-ensemble-suite)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Guidelines

- Follow PEP 8 and PEP 484 (Type Hinting)
- Use Google Python Style Guide for formatting and documentation
- Apply Google Docstrings for all modules, classes, and functions
- Write tests for new features
- Format code using black (configured in pyproject.toml)
- Run type checking with mypy

[Back to top](#ai-ensemble-suite)

## Running Tests

### 👉 Note: Tests are currently broken. They were originally developed early on in the development process and are now out of date. I will be updating these later.
 
```bash
# Install test dependencies
pip install ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=ai_ensemble_suite
```

[Back to top](#ai-ensemble-suite)

## Roadmap

- Refinement and cleanup of the current codebase
- Rewriting the existing tests due to API changes from early code
- Adding OpenAI API compatibility for local LLM servers like LM Studio, Ollama
- Support for Internet API providers

[Back to top](#ai-ensemble-suite)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Special Thanks
🙏 Special thanks to Georgi Gerganov and the whole team working on llama.cpp for making all of this possible.


[Back to top](#ai-ensemble-suite)

