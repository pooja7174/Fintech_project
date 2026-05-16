"""Utility: PDF Report Generator (backward compat shim)"""
from modules.portfolio import _generate_pdf_report

def generate_pdf_report(dataset, accuracy):
    return _generate_pdf_report(dataset, accuracy, [
        "Portfolio Overview", "Model Performance", "Risk Analysis",
        "Stress Test Results", "Compliance Summary"
    ])
