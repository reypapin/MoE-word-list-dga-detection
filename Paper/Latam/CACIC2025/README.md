# CACIC 2025 Conference Submission

This directory contains the LaTeX source files and materials for the paper submitted to CACIC 2025 (Computer Science and Operational Research Congress).

## Paper Information

### Title
**"Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach"**

### Authors
- **Reynier Leyva La O** - GridTICs, Facultad Regional Mendoza, Universidad Tecnológica Nacional, Mendoza, Argentina
- **Rodrigo Gonzalez** - GridTICs, Facultad Regional Mendoza, Universidad Tecnológica Nacional, Mendoza, Argentina  
- **Carlos A. Catania** - LABSIN, Facultad de Ingeniería, Universidad Nacional de Cuyo, Mendoza, Argentina

### Conference
- **Event**: CACIC 2025 (Computer Science and Operational Research Congress)
- **Format**: LNCS (Lecture Notes in Computer Science) style
- **Submission Status**: Conference submission

## Files Description

### Main LaTeX Document
- **`typeinst.tex`**: Main paper source file containing the complete manuscript
- **Format**: LLNCS template compliance
- **Content**: Complete research paper with abstract, introduction, methodology, results, and conclusions

### Bibliography and References
- **`cas-refs.bib`**: BibTeX bibliography file with all paper references
- **References**: Comprehensive citation database for DGA detection research
- **Standards**: ACM/IEEE citation formatting

### LaTeX Class and Style Files
- **`llncs.cls`**: Springer LNCS document class
- **`llncs.dem`**: LNCS demonstration file
- **`llncsdoc.pdf`**: LNCS documentation
- **`llncsdoc.sty`**: LNCS documentation style
- **`splncs03.bst`**: Bibliography style for LNCS
- **`sprmindx.sty`**: Springer index style
- **`typeinst.pdf`**: Sample LNCS paper format

### Supporting Files
- **`history.txt`**: Version history and changes log
- **`readme.txt`**: Basic compilation instructions
- **`eijkel2.eps`**: Example figure (if applicable)

### Figures and Visualizations
- **`Train.jpg`**: Training phase diagram
- **`comparison.pdf`**: Model performance comparison chart
- **`evaluation_phase.pdf`**: Evaluation methodology flowchart
- **`family_heatmap.pdf`**: DGA family characteristic heatmap
- **`performance_comparison.pdf`**: Comparative performance visualization
- **`training_phase.pdf`**: Training process diagram

## Paper Structure

### Abstract
The paper addresses expert model selection for wordlist-based DGA detection within MoE architectures, evaluating seven candidate models and identifying ModernBERT as the optimal expert.

### Key Sections
1. **Introduction**: Problem motivation and MoE paradigm for DGA detection
2. **Related Work**: Current state of DGA detection research and limitations
3. **Problem Formulation**: Mathematical formulation of expert selection challenge
4. **Methodology**: Systematic evaluation framework and experimental design
5. **Experimental Setup**: Dataset, models, and evaluation protocol
6. **Results**: Comprehensive performance analysis across models and families
7. **Discussion**: Analysis of findings and practical implications
8. **Conclusion**: Summary of contributions and future work

### Main Contributions
1. Systematic evaluation framework for MoE expert selection
2. Comprehensive empirical analysis of seven state-of-the-art models
3. Composite performance metric integrating multiple evaluation dimensions
4. Practical deployment guidelines for production systems
5. Performance characterization identifying ModernBERT as optimal expert

## Key Results Presented

### Optimal Expert Model
- **ModernBERT**: 86.7% F1-score on known families, 80.9% on unknown families
- **Inference Time**: 26ms on NVIDIA Tesla T4 GPU
- **Throughput**: ~38,000 domains/second
- **Improvement**: 9.4% better on known families, 30.2% on unknown families

### Model Comparison Table
| Model | Known F1 | Unknown F1 | Inference Time | Throughput |
|-------|----------|------------|----------------|------------|
| ModernBERT | 86.7% | 80.9% | 26ms | 38k/s |
| Gemma 3B | 82.1% | 75.3% | 650ms | 1.5k/s |
| Llama 3.2 3B | 81.4% | 74.8% | 680ms | 1.4k/s |
| CNN | 78.9% | 72.1% | 15ms | 66k/s |
| Random Forest | 75.2% | 68.4% | 5ms | 200k/s |

## Compilation Instructions

### Prerequisites
```bash
# Install LaTeX distribution (e.g., TeX Live)
sudo apt-get install texlive-full

# Or minimal installation
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### Compilation Process
```bash
# Navigate to paper directory
cd Paper/Latam/CACIC2025/

# Compile the paper
pdflatex typeinst.tex
bibtex typeinst
pdflatex typeinst.tex
pdflatex typeinst.tex

# Generated output: typeinst.pdf
```

### Alternative Build
```bash
# Single command compilation
latexmk -pdf typeinst.tex
```

## Figures and Tables

### Figure Sources
All figures included in the paper are generated from the experimental results in this repository:
- Performance comparison charts from `Result_csv/` data
- Training/evaluation flowcharts documenting methodology
- Heatmaps showing DGA family characteristics
- Comparative analysis visualizations

### Table Data
Tables in the paper are derived from:
- Model performance summaries in `Result_csv/`
- Detailed evaluation results in `Result_File/`
- Computational benchmarks from experimental evaluation

## Research Impact

### Novel Contributions
1. **First systematic study** of expert selection for wordlist-based DGA detection
2. **Comprehensive evaluation** of modern ML approaches on challenging DGA families
3. **Practical guidelines** for deploying MoE systems in cybersecurity
4. **Reproducible benchmark** with open-source implementation

### Practical Applications
- Real-time DNS monitoring systems
- Cybersecurity threat intelligence platforms
- Network security infrastructure
- Edge computing deployment scenarios

## Submission Timeline

### Development Phases
1. **Research Phase**: Experimental design and execution
2. **Analysis Phase**: Results compilation and interpretation
3. **Writing Phase**: Manuscript preparation and revision
4. **Submission Phase**: Conference submission and review

### Version Control
- Paper revisions tracked in `history.txt`
- Continuous updates based on experimental results
- Peer review incorporation and refinements

## Related Work Context

### Research Positioning
This work addresses gaps identified in recent DGA detection surveys:
- Limited focus on wordlist-based families in existing literature
- Lack of systematic expert selection methodologies
- Insufficient evaluation of generalization capabilities
- Missing practical deployment considerations

### Innovation Areas
- MoE paradigm application to cybersecurity
- Specialized expert training strategies
- Comprehensive evaluation protocols
- Production-ready performance benchmarks

## Citation Information

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={Proceedings of CACIC 2025},
  year={2024},
  publisher={Springer},
  series={Communications in Computer and Information Science}
}
```

## Conference Information

### CACIC 2025
- **Full Name**: Computer Science and Operational Research Congress
- **Focus**: Computer Science, Information Systems, and Operational Research
- **Format**: International conference with peer review
- **Publication**: Proceedings published by Springer CCIS series
- **Scope**: Theoretical and applied research in computer science

### Submission Requirements
- **Format**: LNCS style, maximum pages as per conference guidelines
- **Language**: English
- **Review Process**: Double-blind peer review
- **Acceptance Criteria**: Technical contribution, novelty, and experimental validation