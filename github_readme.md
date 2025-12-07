# Master Data Management - Contact Deduplication Project

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Project Overview

A comprehensive Master Data Management (MDM) analysis demonstrating data quality assessment, entity resolution, and golden record creation for contact/candidate databases. This project showcases end-to-end MDM capabilities including data profiling, duplicate detection, data enrichment, and actionable insights.

**Perfect for:** MDM Analyst, Data Analyst, Data Quality Analyst positions

## 🎯 Business Problem

Organizations managing contact databases across multiple systems (CRM, ATS, LinkedIn, manual entry) face:
- **Duplicate records** causing operational inefficiency
- **Inconsistent data quality** impacting business decisions
- **Lack of single source of truth** for master data
- **Manual effort** required for data reconciliation

## 💡 Solution

This project demonstrates a systematic approach to Master Data Management:

1. **Data Profiling** - Assess data quality and identify issues
2. **Quality Validation** - Apply business rules and validation logic
3. **Entity Resolution** - Detect and merge duplicate records
4. **Golden Record Creation** - Apply survivorship rules to create master records
5. **Data Enrichment** - Add business value through segmentation and scoring
6. **Actionable Insights** - Provide recommendations for data governance

## 🔧 Technical Skills Demonstrated

### Core MDM Capabilities
- ✅ Data profiling and quality assessment
- ✅ Duplicate detection using standardization and matching algorithms
- ✅ Entity resolution and golden record creation
- ✅ Survivorship rules implementation
- ✅ Data enrichment and business rule application
- ✅ KPI tracking and monitoring

### Technical Stack
- **Python** - Data manipulation and analysis
- **Pandas** - Data processing and transformation
- **NumPy** - Statistical calculations
- **Matplotlib/Seaborn** - Data visualization
- **SQL-ready logic** - Filtering, aggregation, joins

### Data Quality Dimensions
- Completeness
- Validity
- Consistency
- Accuracy
- Timeliness

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Total Source Records** | 500 |
| **Master Records Created** | ~400 |
| **Deduplication Rate** | ~20% |
| **Data Completeness** | 85%+ |
| **Email Validity Rate** | 90%+ |
| **High-Value Contacts Identified** | 100+ |

### Business Impact
- ⏱️ **Estimated 8-10 hours saved monthly** from automated deduplication
- 📈 **Improved contact reachability** through data validation
- 🎯 **Enhanced targeting** through contact segmentation
- 📊 **Better reporting accuracy** with master data

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mdm-contact-deduplication-project.git
cd mdm-contact-deduplication-project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python mdm_analysis.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook mdm_analysis.ipynb
```

## 📁 Project Structure

```
mdm-contact-deduplication-project/
│
├── README.md                      # Project documentation
├── mdm_analysis.ipynb            # Main Jupyter notebook
├── mdm_analysis.py               # Python script version
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
│
├── images/                       # Visualization outputs
│   ├── quality_dashboard.png
│   └── duplicate_analysis.png
│
└── data/                         # Data directory
    └── README.md                 # Note about synthetic data
```

## 🔍 Analysis Sections

### 1. Data Profiling & Quality Assessment
- Missing data analysis
- Data type validation
- Completeness scoring
- Source system distribution

### 2. Data Quality Rules & Validation
- Email format validation
- Phone number validation
- Name completeness checks
- Quality score calculation per record

### 3. Duplicate Detection & Entity Resolution
- Name standardization
- Matching key generation
- Duplicate group identification
- Email-based duplicate detection

### 4. Golden Record Creation
- Survivorship rules implementation:
  - Most recent for temporal data
  - Most complete for missing values
  - Source system hierarchy
- Confidence score calculation

### 5. Data Enrichment
- Engagement score calculation
- Contact segmentation (High/Medium/Low Value, At Risk)
- Enrichment gap identification

### 6. Visualizations & KPIs
- Missing data heatmaps
- Quality score distributions
- Entity resolution metrics
- Segment analysis
- Data quality dashboard

## 📈 Sample Visualizations

*Note: Visualizations are generated when running the script*

The analysis produces a comprehensive dashboard including:
- Missing data patterns
- Quality score distribution
- Source system breakdown
- Duplicate detection results
- Contact segmentation
- Key metrics summary

## 🎓 Key Learnings & Insights

### MDM Best Practices Applied
1. **Standardization First** - Clean and standardize data before matching
2. **Multi-dimensional Matching** - Use multiple attributes for entity resolution
3. **Survivorship Rules** - Clear hierarchy for data conflicts
4. **Confidence Scoring** - Track quality of master records
5. **Continuous Monitoring** - Implement KPIs for ongoing data quality

### Recommendations for Production Implementation
- Implement validation at point of data entry
- Establish automated duplicate detection workflows
- Define clear data stewardship roles
- Schedule regular data quality monitoring
- Create data quality scorecards for stakeholders

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

## 💼 Relevance to MDM Roles

This project demonstrates skills directly applicable to:
- **Master Data Management Analyst**
- **Data Quality Analyst**
- **Data Steward**
- **Data Analyst**

### Tamr Platform Relevance
While this project uses Python, the concepts directly translate to Tamr:
- Data profiling = Tamr's data source analysis
- Duplicate detection = Tamr's machine learning matching
- Golden records = Tamr's mastered records
- Quality rules = Tamr's validation framework

## 🔗 Connect With Me

- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: your.email@example.com
- **Portfolio**: [Your Portfolio Website]

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Synthetic data generated for demonstration purposes
- Project created as part of data analyst portfolio
- Inspired by real-world MDM challenges in recruitment and CRM systems

---

**Note**: This is a portfolio project using synthetic data. All data quality issues and duplicates are artificially generated to demonstrate MDM capabilities.