# Subscription Account Sharing Detection

A machine learning project that uses **K-Means clustering** to identify potentially abusive subscription account usage patterns. The model generates synthetic user activity data and groups accounts into three categories: **Safe**, **Potential Abuse**, and **Abusive**.

## Overview

Subscription account sharing can be identified by analyzing unusual patterns in how an account is accessed and used.

This project applies an **unsupervised learning approach** using K-Means clustering to discover these patterns without requiring a pre-labeled dataset.

Synthetic data is generated to simulate different account usage behaviors, after which the accounts are clustered based on their usage characteristics.

## Account Categories

- 🟢 **Safe** — Normal and consistent usage patterns.
- 🟡 **Potential Abuse** — Some unusual or suspicious activity.
- 🔴 **Abusive** — Strong indicators of account sharing or misuse.

## Features

The synthetic dataset includes behavioral features such as:

- Number of unique devices
- Number of unique locations
- Concurrent sessions
- Login frequency
- Geographic distance between logins
- Average session duration

These features are used to identify similarities and differences between account usage patterns.

## Methodology

```text
Synthetic Data Generation
          ↓
    Data Preprocessing
          ↓
     Feature Scaling
          ↓
    K-Means Clustering
          ↓
  Cluster Interpretation
          ↓
 Safe / Potential Abuse / Abusive
