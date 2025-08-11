# An Empirical Study on Method-Level Performance Evolution in Open-Source Java Projects

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]() [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the replication package for our empirical study analyzing performance evolution patterns in Java projects at the method level. Our research provides comprehensive insights into how code changes impact performance across 15 mature open-source Java projects.

## 📖 Abstract

Performance is a critical quality attribute in software development, yet the impact of method-level code changes on performance evolution remains poorly understood. We conducted a large-scale empirical study analyzing performance evolution in 15 mature open-source Java projects hosted on GitHub. Our analysis encompassed **739 commits** containing **1,499 method-level code changes**, using Java Microbenchmark Harness (JMH) for precise performance measurement and rigorous statistical analysis to quantify both the significance and magnitude of performance variations.

## 🎯 Research Questions

Our study addresses four key research questions:

- **RQ1**: What are the patterns of performance changes in Java projects over time?
- **RQ2**: How do different types of code changes correlate with performance impacts?
- **RQ3**: How do developer experience and code change complexity relate to performance impact magnitude?
- **RQ4**: Are there significant differences in performance evolution patterns across different domains or project sizes?

## 🔍 Key Findings

- **32.7%** of method-level changes result in measurable performance impacts
- Performance regressions occur **1.3 times** more frequently than improvements (18.5% vs 14.2%)
- **No significant differences** in performance impact distributions across code change categories
- **Algorithmic changes** demonstrate the highest improvement potential (25.6%) but carry substantial regression risk (33.9%)
- **Senior developers** produce more stable changes with fewer extreme variations
- **Domain-size interactions** reveal significant patterns, with web server + small projects exhibiting the highest performance instability (42.2%)

## 🏗️ Repository Structure

```
├── jperfevo/                          # Main analysis package
│   ├── core/                          # Core analysis components
│   │   ├── agreement_analyzer.py      # Inter-rater agreement analysis (Cohen's κ)
│   │   ├── code_diff_generator.py     # Code difference visualization
│   │   ├── code_pair_generator.py     # Method pair extraction from Git history
│   │   ├── code_pair_inserter.py      # Database insertion utilities
│   │   ├── github_author_experience.py # Developer experience quantification
│   │   ├── method_complexity_analyzer.py # Code change complexity scoring
│   │   ├── method_mapper.py           # Method mapping between commit versions
│   │   └── performance_diff_significance.py # Statistical significance testing
│   ├── models/                        # Data models
│   │   └── code_pair.py              # Code pair data structure
│   ├── rq/                           # Research question analysis modules
│   │   ├── rq1.py                    # RQ1: Temporal performance patterns
│   │   ├── rq2.py                    # RQ2: Code change type analysis
│   │   ├── rq3.py                    # RQ3: Developer experience & complexity
│   │   └── rq4.py                    # RQ4: Domain and size analysis
│   └── services/                     # Utility services
│       ├── db_service.py             # MongoDB database operations
│       └── similarity_service.py     # Code similarity analysis
├── jphb-performance-data/             # Raw performance measurement data
│   ├── chronicle-core/               # Performance data per project
│   ├── client-java/                  
│   ├── objenesis/                    
│   └── protostuff/                   
│       └── performance_data.json     # JMH benchmark execution results
├── results/                          # Processed analysis results
│   ├── [project-name]/              # Per-project analysis results
│   │   ├── author_experiences.json   # Developer experience scores
│   │   ├── method_complexities.json  # Code change complexity metrics
│   │   ├── method_mappings.json      # Method version mappings
│   │   └── labelings.json           # Code change type classifications
│   ├── apm-agent-java/              # 15 projects total with results
│   ├── chronicle-core/              
│   ├── client-java/                 
│   ├── feign/                       
│   ├── hdrhistogram/                
│   ├── jctools/                     
│   ├── jdbi/                        
│   ├── jersey/                      
│   ├── jetty/                       
│   ├── netty/                       
│   ├── objenesis/                   
│   ├── protostuff/                  
│   ├── simpleflatmapper/            
│   └── zipkin/                      
├── projects.json                     # Study dataset configuration
├── statistics.json                   # Project statistics and metadata
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 📊 Dataset

Our study analyzes **15 mature open-source Java projects** across diverse domains:

| Project | Domain | KLOC | Method Changes | Benchmarks | Results Available |
|---------|--------|------|----------------|------------|-------------------|
| **jetty** | Web Server | 339.06 | 2,472 | 12,720 | ✅ |
| **netty** | Networking | 216.98 | 4,241 | 7,669 | ✅ |
| **fastjson2** | Data Processing | 178.5 | 1,726 | 3,725 | ✅ |
| **apm-agent-java** | Monitoring | 80.22 | 891 | 2,984 | ✅ |
| **jersey** | Web Server | 158.91 | 310 | 781 | ✅ |
| **simpleflatmapper** | Data Processing | 51.79 | 911 | 1,969 | ✅ |
| **protostuff** | Data Processing | 42.29 | 448 | 1,354 | ✅ |
| **jctools** | System Programming | 31.48 | 339 | 1,042 | ✅ |
| **jdbi** | Data Processing | 28.49 | 1,266 | 1,919 | ✅ |
| **client-java** | Monitoring | 27.38 | 155 | 667 | ✅ |
| **zipkin** | Monitoring | 23.51 | 656 | 2,726 | ✅ |
| **feign** | Web Server | 17.42 | 351 | 1,384 | ✅ |
| **chronicle-core** | System Programming | 13.25 | 780 | 3,170 | ✅ |
| **hdrhistogram** | Monitoring | 8.89 | 158 | 317 | ✅ |
| **objenesis** | Testing | 2.69 | 107 | 784 | ✅ |

**Total**: 1,499 method-level changes across 739 commits

### Data Completeness

This replication package includes **complete processed datasets**, making it immediately usable for:
- ✅ **Verification of statistical analyses** without recomputation
- ✅ **Extension studies** using our methodology on new projects  
- ✅ **Comparative analyses** across different project characteristics
- ✅ **Educational use** for performance analysis techniques

All results in the `results/` directory are derived from **~80 machine days** of computation, providing reviewers with immediate access to the complete study dataset.

## 🚀 Setup and Installation

### Prerequisites

- **Python 3.8+**
- **Java 8+** (for benchmark execution)
- **MongoDB** (for data storage)
- **Git** (for repository cloning)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mooselab/empirical-java-performance-evolution
cd empirical-java-performance-evolution
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
DB_NAME=your_db_name
DB_URL=your_mongodb_url
CLOUD_DB_URL=your_cloud_mongodb_url  # Optional
GITHUB_TOKEN=your_github_token       # For API access
```

## 📈 Usage

### Quick Start with Processed Data

The repository includes processed results for immediate analysis. You can directly run research question analyses on the provided datasets:

```bash
# Analyze temporal performance patterns using processed data
python -m jperfevo.rq.rq1

# Examine code change type impacts
python -m jperfevo.rq.rq2

# Explore developer experience and complexity relationships
python -m jperfevo.rq.rq3

# Investigate domain and size patterns
python -m jperfevo.rq.rq4
```

## 🔬 Methodology

### Performance Measurement

- **Microbenchmarking**: Java Microbenchmark Harness (JMH) with 3 forks and 5 iterations (15 iterations in total)
- **Instrumentation**: Custom Java bytecode instrumentation for method-level metrics (see [JIB](https://github.com/kavehshahedi/java-instrumentation-buddy))
- **Statistical Analysis**: Mann-Whitney U-test (p < 0.05) and Cliff's Delta effect size (|δ| ≥ 0.147)

### Code Change Classification

We categorize method-level changes into seven types:
- **ALG**: Algorithmic Change
- **CF**: Control Flow
- **DS**: Data Structure & Variable
- **REF**: Refactoring & Code Cleanup
- **ER**: Exception & Return Handling
- **CON**: Concurrency
- **API**: API/Library Call

### Developer Experience Scoring

Multi-dimensional experience quantification based on:
- GitHub account age (20% weight)
- Project-specific contributions (30% weight)  
- Total contributions across projects (25% weight)
- Code review participation (25% weight)

## 📊 Results and Visualizations

The analysis generates comprehensive visualizations and statistics:

- **Temporal patterns** across project lifecycles
- **Code change impact distributions** by category
- **Developer experience correlations** with performance outcomes
- **Domain-specific and size-based patterns**
- **Statistical significance tests** for all major findings

Results are saved in the `plots/` directory organized by research question.

## 🔄 Reproducibility

### Data Availability

This replication package provides **complete processed datasets** including:

- **Study analysis dataset**: The dataset containing all the information aggregately in `dataset/dataset.csv`
- **Performance measurements**: Raw JMH benchmark execution results in `jphb-performance-data/`
- **Method mappings**: Version tracking across 1,499 method changes in `results/*/method_mappings.json`
- **Developer experience scores**: Quantified contributor expertise in `results/*/author_experiences.json`
- **Code change complexity metrics**: Weighted complexity scores in `results/*/method_complexities.json`
- **Code change classifications**: Manual labels with inter-rater agreement (κ = 0.96) in `results/*/labelings.json`
- **Statistical test results**: All significance tests and effect sizes embedded in analysis scripts

### Processed Results Structure

Each project in `results/` contains:
```
project-name/
├── author_experiences.json    # Developer experience quantification
├── method_complexities.json   # Code change complexity analysis  
├── method_mappings.json       # Method version mappings with performance data
└── labelings.json            # Code change type classifications (when available)
```

## 🤝 Contributing

We welcome contributions to improve the analysis tools and extend the study:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -am 'Add improvement'`)
4. **Push** to the branch (`git push origin feature/improvement`)
5. **Create** a Pull Request

## 📧 Contact

- **Kaveh Shahedi** - [kaveh.shahedi@polymtl.ca](mailto:kaveh.shahedi@polymtl.ca)
- **Heng Li** (Corresponding Author) - [heng.li@polymtl.ca](mailto:heng.li@polymtl.ca)

**Department of Computer Engineering and Software Engineering**  
Polytechnique Montréal, Canada

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Natural Sciences and Engineering Research Council of Canada (NSERC)** - Grant #RGPIN-2021-03900
- **Open-source projects** and their maintainers for providing high-quality benchmarks
- **Research community** for foundational work in performance analysis