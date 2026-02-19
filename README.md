# MSEDCL Electricity Tariff Analysis & ML Forecasting

An automated data pipeline and machine learning project designed to extract, clean, analyze, and forecast electricity tariffs for the Maharashtra State Electricity Distribution Co. Ltd. (MSEDCL).

## Overview

Electricity rates are complex and historically buried in difficult-to-read PDF documents. This project solves that problem by scraping the raw PDF data, filtering anomalies, calculating Compound Annual Growth Rates (CAGR), and using Machine Learning predicition algorithm to predict rates up to 2030. 

**Key Features:**
* **PDF Rendered Data Parsing:** Raw data is extracted from MSEDCL tariff orders using `pdfplumber`.
* **Data Sanitization:** Detects and removes column-shift anomalies and outlier artifacts native to OCR/PDF scraping.
* **Scenario-Based ML Forecasting:** Uses Linear Regression combined with compounding trend modifiers to predict three distinct economic futures:
  * *Expected Scenario:* Baseline historical inflation.
  * *Optimistic Scenario:* High renewable efficiency dropping prices (-2% YoY deviation).
  * *Pessimistic Scenario:* Grid strain and fuel shortages (+3% YoY deviation).
* **Automated Visualizations:** Generates high-resolution heatmaps, fan charts, and bar graphs for immediate insights.

## Data Source

All historical data (2009–2025) utilized in this project is publicly available and was downloaded directly from the official Maharashtra Electricity portal:
[MSEDCL Tariff Details](https://www.mahadiscom.in/en/consumer/tariff-details/)
