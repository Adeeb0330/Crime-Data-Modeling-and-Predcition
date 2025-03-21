# Crime Data Modeling & Prediction

## Overview
This project focuses on analyzing and predicting crime trends in India using Python and Machine Learning. The dataset (2020-2024) was cleaned, processed, and transformed to build a predictive model capable of forecasting crime trends for the years 2025-2030.

## Features
- **Data Cleaning & Preprocessing**: Handled missing values, removed inconsistencies, and performed feature engineering.
- **Exploratory Data Analysis (EDA)**: Visualized crime patterns across different regions and time periods.
- **Machine Learning Model**: Built a **RandomForestRegressor** model with an **R² score of 0.93** to predict future crime trends.
- **Crime Trend Forecasting**: Provided insights into potential crime patterns for better law enforcement planning.

## Dataset
- **Time Period**: 2020 - 2024
- **Features**:
  - Crime Type
  - Location
  - Year
  - Crime Description
  - Case Closed Status
  - Other relevant crime attributes
  - 📂 **File:** [crime_data.csv]("crime_dataset_india.csv")  
🔗 **Download:** Click [here](crime_dataset_india.csv) to access the dataset.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Model Used**: RandomForestRegressor

## Results
- Achieved a **93% accuracy (R² score = 0.93)** in predicting future crime rates.
- Identified high-risk locations and specific crime types.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Visualizations
Below are some insights from the data analysis:
- What are the most common crime types?
- How do crime trends change over the years?
- What is the distribution of crimes across different crime types?
- Which crime types have the highest number of reported cases?
- How does the number of crimes vary across different years?

## Future Enhancements
- Integrate a web-based dashboard for real-time crime monitoring.
- Experiment with deep learning models for improved accuracy.
- Automate data collection from government crime records.

## Contributors
- **Adeeb** - [GitHub Profile](https://github.com/Adeeb0330)

## License
This project is open-source and available under the [MIT License](LICENSE).

---
Feel free to contribute or report any issues!
