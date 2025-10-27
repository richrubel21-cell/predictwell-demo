# app_expanded.py
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import os
import uvicorn
import logging
import asyncio
import random
import json

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
APP_VERSION = os.getenv("APP_VERSION", "2025.10.27")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predictwell")

app = FastAPI(
    title="PredictWell Health AI",
    version=APP_VERSION,
    description="Multi-condition predictive monitoring platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace("%", "")
        return float(s)
    except Exception:
        return default

def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"y", "yes", "true", "1", "on"}

def get(payload: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    """Fetch first present key (any casing/spacing/underscore/hyphen)."""
    variants: List[str] = []
    for k in keys:
        variants += [
            k,
            k.lower(),
            k.upper(),
            k.replace(" ", "_"),
            k.replace(" ", "").lower(),
            k.replace("-", "_"),
        ]
    for name in variants:
        if name in payload:
            return to_float(payload[name], default)
    return default

def present(payload: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        if any(name in payload for name in [k, k.lower(), k.upper(), k.replace(" ", "_"), k.replace("-", "_")]):
            return True
    return False

def time_to_event(risk: float) -> str:
    """Convert risk score to estimated time-to-event"""
    if risk >= 90:
        return "< 1 hour"
    elif risk >= 80:
        return "1-2 hours"
    elif risk >= 70:
        return "2-4 hours"
    elif risk >= 60:
        return "4-8 hours"
    elif risk >= 50:
        return "8-24 hours"
    elif risk >= 40:
        return "24-48 hours"
    else:
        return "stable"

def forecast_future(risk: float) -> Dict[str, str]:
    """Generate 3-day, 5-day, 1-week, 2-week forecasts"""
    if risk >= 80:
        return {"3day": "High", "5day": "High", "1week": "Critical", "2week": "Critical"}
    elif risk >= 60:
        return {"3day": "Moderate", "5day": "High", "1week": "High", "2week": "Critical"}
    elif risk >= 40:
        return {"3day": "Low", "5day": "Moderate", "1week": "Moderate", "2week": "High"}
    else:
        return {"3day": "Stable", "5day": "Stable", "1week": "Low", "2week": "Low"}

# -----------------------------------------------------------------------------
# SEPSIS PREDICTION MODULE
# -----------------------------------------------------------------------------
def predict_sepsis(payload: Dict[str, Any]) -> Tuple[float, List[str], str]:
    """
    Sepsis risk prediction based on qSOFA, SIRS, and organ dysfunction indicators.
    Returns: (risk_score, explanations, time_to_event)
    """
    explain: List[str] = []
    risk = 0.0

    # Core vitals - SIRS criteria
    temp = get(payload, ["temperature_c", "temp_c", "temperature"])
    if temp > 38 or (temp > 0 and temp < 36):
        risk += 15
        explain.append("abnormal temperature (SIRS)")

    hr = get(payload, ["hr", "heart_rate"])
    if hr > 90:
        risk += 10
        explain.append("tachycardia (SIRS)")

    rr = get(payload, ["rr", "resp_rate", "respiratory_rate"])
    if rr > 20:
        risk += 10
        explain.append("tachypnea (SIRS)")

    # qSOFA criteria
    sbp = get(payload, ["sbp", "systolic_bp"])
    if sbp > 0 and sbp <= 100:
        risk += 12
        explain.append("hypotension (qSOFA)")

    if rr >= 22:
        risk += 8
        explain.append("respiratory distress (qSOFA)")

    gcs = get(payload, ["gcs", "glasgow_coma_scale"], default=15)
    if gcs < 15:
        risk += 10
        explain.append("altered mentation (qSOFA)")

    # Lab markers
    wbc = get(payload, ["wbc", "white_blood_cell"])
    if wbc > 12 or (wbc > 0 and wbc < 4):
        risk += 12
        explain.append("abnormal WBC")

    lactate = get(payload, ["lactate", "lactic_acid"])
    if lactate >= 4:
        risk += 20
        explain.append("severe hyperlactatemia")
    elif lactate >= 2:
        risk += 12
        explain.append("elevated lactate")

    # Organ dysfunction markers
    creat = get(payload, ["cr", "creatinine"])
    if creat >= 2.0:
        risk += 10
        explain.append("acute kidney injury")

    bilirubin = get(payload, ["bilirubin", "total_bilirubin"])
    if bilirubin >= 2.0:
        risk += 8
        explain.append("hepatic dysfunction")

    platelets = get(payload, ["platelets", "platelet_count"])
    if platelets > 0 and platelets < 100:
        risk += 10
        explain.append("thrombocytopenia")

    # Infection indicators
    if to_bool(payload.get("infection_suspected") or payload.get("infection")):
        risk += 15
        explain.append("suspected infection")

    procalcitonin = get(payload, ["procalcitonin", "pct"])
    if procalcitonin >= 2.0:
        risk += 15
        explain.append("elevated procalcitonin")
    elif procalcitonin >= 0.5:
        risk += 8
        explain.append("possible bacterial infection")

    # Bandemia
    bands = get(payload, ["bands", "band_neutrophils"])
    if bands > 10:
        risk += 8
        explain.append("bandemia")

    # Comorbidities
    if to_bool(payload.get("immunocompromised")):
        risk += 8
        explain.append("immunocompromised")

    if to_bool(payload.get("recent_surgery")):
        risk += 5
        explain.append("post-surgical")

    # Cap risk
    risk = max(0.0, min(100.0, risk))
    
    # Deduplicate explanations
    seen = set()
    uniq = []
    for e in explain:
        if e not in seen:
            uniq.append(e)
            seen.add(e)

    return risk, uniq, time_to_event(risk), forecast_future(risk)

# -----------------------------------------------------------------------------
# RESPIRATORY DETERIORATION MODULE
# -----------------------------------------------------------------------------
def predict_respiratory(payload: Dict[str, Any]) -> Tuple[float, List[str], str]:
    """
    Respiratory failure prediction based on gas exchange, work of breathing, and trends.
    Returns: (risk_score, explanations, time_to_event)
    """
    explain: List[str] = []
    risk = 0.0

    # Oxygenation
    spo2 = get(payload, ["spo2", "oxygen_saturation"], default=98.0)
    fio2 = get(payload, ["fio2", "oxygen_percentage"], default=21.0)
    
    # Calculate P/F ratio approximation if PaO2 available
    pao2 = get(payload, ["pao2", "arterial_oxygen"])
    if pao2 > 0 and fio2 > 0:
        pf_ratio = pao2 / (fio2 / 100.0)
        if pf_ratio < 100:
            risk += 25
            explain.append("severe ARDS (P/F < 100)")
        elif pf_ratio < 200:
            risk += 18
            explain.append("moderate ARDS (P/F < 200)")
        elif pf_ratio < 300:
            risk += 12
            explain.append("mild ARDS (P/F < 300)")

    # SpO2/FiO2 ratio (alternative when no ABG)
    if pao2 == 0 and fio2 > 21:
        sf_ratio = spo2 / (fio2 / 100.0)
        if sf_ratio < 200:
            risk += 18
            explain.append("poor oxygenation (S/F ratio)")

    # Oxygen requirement escalation
    if fio2 >= 60:
        risk += 15
        explain.append("high oxygen requirement")
    elif fio2 >= 40:
        risk += 10
        explain.append("moderate oxygen requirement")

    # Respiratory rate and pattern
    rr = get(payload, ["rr", "resp_rate", "respiratory_rate"])
    if rr > 30:
        risk += 15
        explain.append("severe tachypnea")
    elif rr > 24:
        risk += 10
        explain.append("tachypnea")
    elif rr < 10 and rr > 0:
        risk += 12
        explain.append("respiratory depression")

    # Work of breathing indicators
    if to_bool(payload.get("accessory_muscles") or payload.get("increased_work")):
        risk += 12
        explain.append("increased work of breathing")

    if to_bool(payload.get("retractions")):
        risk += 10
        explain.append("retractions observed")

    # ABG values
    paco2 = get(payload, ["paco2", "arterial_co2"])
    if paco2 >= 50:
        risk += 15
        explain.append("hypercapnia")
    elif paco2 >= 45:
        risk += 8
        explain.append("CO2 retention")

    ph = get(payload, ["ph", "arterial_ph"])
    if ph > 0 and ph < 7.30:
        risk += 15
        explain.append("severe acidosis")
    elif ph > 0 and ph < 7.35:
        risk += 8
        explain.append("acidosis")

    # Lung pathology
    if to_bool(payload.get("infiltrates") or payload.get("pneumonia")):
        risk += 10
        explain.append("pulmonary infiltrates")

    if to_bool(payload.get("copd") or payload.get("asthma")):
        risk += 8
        explain.append("chronic lung disease")

    if to_bool(payload.get("pleural_effusion")):
        risk += 8
        explain.append("pleural effusion")

    # Ventilatory support
    if to_bool(payload.get("mechanical_ventilation") or payload.get("intubated")):
        risk += 15
        explain.append("mechanical ventilation")
    elif to_bool(payload.get("niv") or payload.get("bipap") or payload.get("cpap")):
        risk += 12
        explain.append("non-invasive ventilation")
    elif to_bool(payload.get("high_flow_nasal_cannula") or payload.get("hfnc")):
        risk += 10
        explain.append("high-flow oxygen")

    # Trends (if provided)
    spo2_trend = payload.get("spo2_trend", "").lower()
    if "declining" in spo2_trend or "worsening" in spo2_trend:
        risk += 10
        explain.append("declining oxygenation trend")

    rr_trend = payload.get("rr_trend", "").lower()
    if "increasing" in rr_trend:
        risk += 8
        explain.append("increasing respiratory rate trend")

    # Cap risk
    risk = max(0.0, min(100.0, risk))
    
    # Deduplicate
    seen = set()
    uniq = []
    for e in explain:
        if e not in seen:
            uniq.append(e)
            seen.add(e)

    return risk, uniq, time_to_event(risk), forecast_future(risk)

# -----------------------------------------------------------------------------
# CARDIAC EVENT PREDICTION MODULE
# -----------------------------------------------------------------------------
def predict_cardiac(payload: Dict[str, Any]) -> Tuple[float, List[str], str]:
    """
    Cardiac event prediction (MI, arrest, arrhythmia, heart failure decompensation).
    Returns: (risk_score, explanations, time_to_event)
    """
    explain: List[str] = []
    risk = 0.0

    # Hemodynamics
    hr = get(payload, ["hr", "heart_rate"])
    if hr > 120:
        risk += 12
        explain.append("severe tachycardia")
    elif hr > 100:
        risk += 8
        explain.append("tachycardia")
    elif hr < 50 and hr > 0:
        risk += 10
        explain.append("bradycardia")

    sbp = get(payload, ["sbp", "systolic_bp"])
    dbp = get(payload, ["dbp", "diastolic_bp"])
    
    if sbp > 0 and dbp > 0:
        map_est = dbp + (sbp - dbp) / 3.0
        if map_est < 60:
            risk += 20
            explain.append("severe hypotension")
        elif map_est < 65:
            risk += 12
            explain.append("hypotension")

    if sbp > 180:
        risk += 10
        explain.append("severe hypertension")

    # Cardiac biomarkers
    troponin = get(payload, ["troponin", "troponin_i", "troponin_t"])
    if troponin >= 1.0:
        risk += 25
        explain.append("elevated troponin (myocardial injury)")
    elif troponin >= 0.1:
        risk += 15
        explain.append("elevated troponin")

    bnp = get(payload, ["bnp", "ntprobnp", "nt_pro_bnp"])
    if bnp >= 1000:
        risk += 15
        explain.append("elevated BNP (heart failure)")
    elif bnp >= 400:
        risk += 10
        explain.append("moderately elevated BNP")

    # ECG findings
    if to_bool(payload.get("st_elevation") or payload.get("stemi")):
        risk += 30
        explain.append("ST elevation (STEMI)")
    elif to_bool(payload.get("st_depression")):
        risk += 20
        explain.append("ST depression")
    elif to_bool(payload.get("t_wave_inversion")):
        risk += 12
        explain.append("T wave inversion")

    if to_bool(payload.get("new_lbbb") or payload.get("left_bundle_branch_block")):
        risk += 15
        explain.append("new LBBB")

    # Arrhythmias
    if to_bool(payload.get("vfib") or payload.get("ventricular_fibrillation")):
        risk += 35
        explain.append("ventricular fibrillation")
    elif to_bool(payload.get("vtach") or payload.get("ventricular_tachycardia")):
        risk += 30
        explain.append("ventricular tachycardia")
    elif to_bool(payload.get("afib") or payload.get("atrial_fibrillation")):
        risk += 12
        explain.append("atrial fibrillation")
    elif to_bool(payload.get("svt") or payload.get("supraventricular_tachycardia")):
        risk += 10
        explain.append("supraventricular tachycardia")

    # Heart failure indicators
    if to_bool(payload.get("pulmonary_edema") or payload.get("rales")):
        risk += 12
        explain.append("pulmonary edema")

    if to_bool(payload.get("jugular_venous_distension") or payload.get("jvd")):
        risk += 8
        explain.append("volume overload")

    if to_bool(payload.get("peripheral_edema")):
        risk += 5
        explain.append("peripheral edema")

    # Chest pain
    chest_pain = get(payload, ["chest_pain_severity", "chest_pain"], default=0)
    if chest_pain >= 7:
        risk += 15
        explain.append("severe chest pain")
    elif chest_pain >= 4:
        risk += 10
        explain.append("chest pain")

    # Cardiac history
    if to_bool(payload.get("prior_mi") or payload.get("history_mi")):
        risk += 8
        explain.append("prior MI")

    if to_bool(payload.get("chf") or payload.get("heart_failure")):
        risk += 10
        explain.append("heart failure history")

    if to_bool(payload.get("cad") or payload.get("coronary_artery_disease")):
        risk += 8
        explain.append("coronary artery disease")

    # Ischemia risk factors
    if to_bool(payload.get("diabetes")):
        risk += 5
        explain.append("diabetes (cardiac risk)")

    age = get(payload, ["age_years", "age"])
    if age >= 75:
        risk += 5
        explain.append("advanced age")

    # Perfusion
    lactate = get(payload, ["lactate"])
    if lactate >= 4:
        risk += 12
        explain.append("poor perfusion (lactate)")
    elif lactate >= 2:
        risk += 6
        explain.append("elevated lactate")

    # Medications/interventions
    if to_bool(payload.get("pressors") or payload.get("vasopressors")):
        risk += 15
        explain.append("vasopressor support")

    if to_bool(payload.get("cardiac_arrest_history")):
        risk += 12
        explain.append("prior cardiac arrest")

    # Trends
    hr_trend = payload.get("hr_trend", "").lower()
    if "increasing" in hr_trend:
        risk += 8
        explain.append("worsening tachycardia")

    bp_trend = payload.get("bp_trend", "").lower()
    if "declining" in bp_trend or "dropping" in bp_trend:
        risk += 10
        explain.append("declining blood pressure")

    # Cap risk
    risk = max(0.0, min(100.0, risk))
    
    # Deduplicate
    seen = set()
    uniq = []
    for e in explain:
        if e not in seen:
            uniq.append(e)
            seen.add(e)

    return risk, uniq, time_to_event(risk), forecast_future(risk)

# -----------------------------------------------------------------------------
# FALL RISK (original eldercare module, simplified)
# -----------------------------------------------------------------------------
def predict_falls(payload: Dict[str, Any]) -> Tuple[float, List[str], str]:
    """
    Fall risk prediction.
    Returns: (risk_score, explanations, time_to_event)
    """
    explain: List[str] = []
    risk = 0.0

    # Age
    age = get(payload, ["age_years", "age"])
    if age >= 85:
        risk += 12
        explain.append("advanced age")
    elif age >= 75:
        risk += 8
        explain.append("age 75-84")

    # Fall history
    falls_12m = get(payload, ["falls_12m", "falls_last_year"])
    if falls_12m >= 2:
        risk += 15
        explain.append("recurrent falls")
    elif falls_12m == 1:
        risk += 10
        explain.append("recent fall")

    # Mobility
    if to_bool(payload.get("unsteady") or payload.get("gait_unsteady")):
        risk += 12
        explain.append("unsteady gait")

    if to_bool(payload.get("walker") or payload.get("assistive_device")):
        risk += 8
        explain.append("assistive device use")

    # Cognitive
    if to_bool(payload.get("delirium_risk") or payload.get("delirium")):
        risk += 15
        explain.append("delirium")

    if to_bool(payload.get("cognition_issues") or payload.get("dementia")):
        risk += 10
        explain.append("cognitive impairment")

    # Medications
    med_count = get(payload, ["med_count", "medications_count"])
    if med_count >= 10:
        risk += 10
        explain.append("polypharmacy")

    if to_bool(payload.get("sedatives") or payload.get("benzodiazepines")):
        risk += 12
        explain.append("sedating medications")

    # Orthostatic hypotension
    if present(payload, ["orthostatic_drop"]):
        drop = get(payload, ["orthostatic_drop"], default=0)
        if drop >= 20:
            risk += 10
            explain.append("orthostatic hypotension")

    # Vision/hearing
    if to_bool(payload.get("vision_impairment")):
        risk += 8
        explain.append("vision impairment")

    # Environment
    if to_bool(payload.get("home_hazards")):
        risk += 5
        explain.append("environmental hazards")

    # Cap risk
    risk = max(0.0, min(100.0, risk))
    
    # Deduplicate
    seen = set()
    uniq = []
    for e in explain:
        if e not in seen:
            uniq.append(e)
            seen.add(e)

    return risk, uniq, time_to_event(risk), forecast_future(risk)
# -----------------------------------------------------------------------------
# UNIFIED PREDICTION ENDPOINT
# -----------------------------------------------------------------------------
class MultiRiskResponse(BaseModel):
    status: str
    patient_id: str
    timestamp: str
    predictions: Dict[str, Dict[str, Any]]
    overall_severity: str
    version: str

@app.post("/predict/all", response_model=MultiRiskResponse)
def predict_all_conditions(payload: Dict[str, Any] = Body(...)):
    """
    Run all prediction modules and return comprehensive risk assessment.
    """
    patient_id = payload.get("patient_id", "unknown")
    
    # Run all predictors
    sepsis_risk, sepsis_explain, sepsis_time, sepsis_forecast = predict_sepsis(payload)
resp_risk, resp_explain, resp_time, resp_forecast = predict_respiratory(payload)
cardiac_risk, cardiac_explain, cardiac_time, cardiac_forecast = predict_cardiac(payload)
fall_risk, fall_explain, fall_time, fall_forecast = predict_falls(payload)
    # Determine overall severity
    max_risk = max(sepsis_risk, resp_risk, cardiac_risk, fall_risk)
    if max_risk >= 80:
        overall = "CRITICAL"
    elif max_risk >= 60:
        overall = "HIGH"
    elif max_risk >= 40:
        overall = "MODERATE"
    else:
        overall = "LOW"

    predictions = {
        "sepsis": {
            "risk_score": round(sepsis_risk, 1),
            "time_to_event": sepsis_time,
            "explanations": sepsis_explain[:8],
            "severity": "CRITICAL" if sepsis_risk >= 80 else "HIGH" if sepsis_risk >= 60 else "MODERATE" if sepsis_risk >= 40 else "LOW"
        },
        "respiratory": {
            "risk_score": round(resp_risk, 1),
            "time_to_event": resp_time,
            "explanations": resp_explain[:8],
            "severity": "CRITICAL" if resp_risk >= 80 else "HIGH" if resp_risk >= 60 else "MODERATE" if resp_risk >= 40 else "LOW"
        },
        "cardiac": {
            "risk_score": round(cardiac_risk, 1),
            "time_to_event": cardiac_time,
            "explanations": cardiac_explain[:8],
            "severity": "CRITICAL" if cardiac_risk >= 80 else "HIGH" if cardiac_risk >= 60 else "MODERATE" if cardiac_risk >= 40 else "LOW"
        },
        "falls": {
            "risk_score": round(fall_risk, 1),
            "time_to_event": fall_time,
            "explanations": fall_explain[:8],
            "severity": "CRITICAL" if fall_risk >= 80 else "HIGH" if fall_risk >= 60 else "MODERATE" if fall_risk >= 40 else "LOW"
        }
    }

    return MultiRiskResponse(
        status="ok",
        patient_id=patient_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        predictions=predictions,
        overall_severity=overall,
        version=APP_VERSION
    )

# -----------------------------------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------------------------------
class PatientSimulator:
    """Generates realistic patient deterioration scenarios"""
    
    def __init__(self, patient_id: str, scenario: str = "stable"):
        self.patient_id = patient_id
        self.scenario = scenario  # stable, sepsis, respiratory, cardiac, falls
        self.start_time = datetime.utcnow()
        self.elapsed_hours = 0.0
        
        # Patient demographics
        self.age = random.randint(55, 90)
        self.name = self._generate_name()
        
        # Initialize baseline vitals
        self.vitals = self._init_baseline()
        
    def _generate_name(self) -> str:
        first_names = ["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", 
                      "William", "Linda", "David", "Barbara", "Richard", "Susan"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                     "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _init_baseline(self) -> Dict[str, float]:
        """Initialize stable baseline vitals"""
        return {
            "hr": random.uniform(65, 85),
            "sbp": random.uniform(110, 135),
            "dbp": random.uniform(65, 85),
            "rr": random.uniform(14, 18),
            "spo2": random.uniform(95, 99),
            "temperature_c": random.uniform(36.5, 37.2),
            "lactate": random.uniform(0.5, 1.5),
            "wbc": random.uniform(5, 10),
            "creatinine": random.uniform(0.7, 1.2),
            "troponin": random.uniform(0.001, 0.01),
            "glucose": random.uniform(80, 120),
            "fio2": 21.0,
        }
    
    def advance_time(self, hours: float):
        """Advance simulation time and evolve vitals based on scenario"""
        self.elapsed_hours += hours
        
        if self.scenario == "stable":
            self._evolve_stable()
        elif self.scenario == "sepsis":
            self._evolve_sepsis()
        elif self.scenario == "respiratory":
            self._evolve_respiratory()
        elif self.scenario == "cardiac":
            self._evolve_cardiac()
        elif self.scenario == "falls":
            self._evolve_falls()
    
    def _add_noise(self, value: float, noise_pct: float = 2.0) -> float:
        """Add realistic measurement noise"""
        noise = random.uniform(-noise_pct/100, noise_pct/100)
        return value * (1 + noise)
    
    def _evolve_stable(self):
        """Stable patient - minor fluctuations only"""
        # Add small random walk
        self.vitals["hr"] += random.uniform(-0.5, 0.5)
        self.vitals["sbp"] += random.uniform(-1, 1)
        self.vitals["rr"] += random.uniform(-0.1, 0.1)
        
        # Keep within normal bounds
        self.vitals["hr"] = max(60, min(90, self.vitals["hr"]))
        self.vitals["sbp"] = max(100, min(140, self.vitals["sbp"]))
        self.vitals["rr"] = max(12, min(20, self.vitals["rr"]))
    
    def _evolve_sepsis(self):
        """Septic patient - progressive deterioration"""
        # Progressive tachycardia
        if self.elapsed_hours < 2:
            self.vitals["hr"] += 1.5
        elif self.elapsed_hours < 6:
            self.vitals["hr"] += 2.5
        else:
            self.vitals["hr"] += 3.0
        
        # Progressive tachypnea
        if self.elapsed_hours > 3:
            self.vitals["rr"] += 0.8
        
        # Hypotension develops
        if self.elapsed_hours > 4:
            self.vitals["sbp"] -= 2.0
        
        # Rising temperature
        if self.elapsed_hours < 8:
            self.vitals["temperature_c"] += 0.15
        
        # Rising lactate
        if self.elapsed_hours > 2:
            self.vitals["lactate"] += 0.25
        
        # Rising WBC
        if self.elapsed_hours > 4:
            self.vitals["wbc"] += 0.5
        
        # Caps
        self.vitals["hr"] = min(140, self.vitals["hr"])
        self.vitals["rr"] = min(32, self.vitals["rr"])
        self.vitals["sbp"] = max(75, self.vitals["sbp"])
        self.vitals["temperature_c"] = min(40.0, self.vitals["temperature_c"])
        self.vitals["lactate"] = min(8.0, self.vitals["lactate"])
        self.vitals["wbc"] = min(25, self.vitals["wbc"])
    
    def _evolve_respiratory(self):
        """Respiratory failure - hypoxia and increased work of breathing"""
        # Declining SpO2
        if self.elapsed_hours < 2:
            self.vitals["spo2"] -= 0.3
        elif self.elapsed_hours < 6:
            self.vitals["spo2"] -= 0.6
        else:
            self.vitals["spo2"] -= 1.0
        
        # Increasing RR
        if self.elapsed_hours > 1:
            self.vitals["rr"] += 0.7
        
        # Tachycardia
        if self.elapsed_hours > 3:
            self.vitals["hr"] += 1.5
        
        # FiO2 escalation (treatment response)
        if self.vitals["spo2"] < 92 and self.vitals["fio2"] < 100:
            self.vitals["fio2"] = min(100, self.vitals["fio2"] + 5)
        
        # Caps
        self.vitals["spo2"] = max(82, self.vitals["spo2"])
        self.vitals["rr"] = min(35, self.vitals["rr"])
        self.vitals["hr"] = min(130, self.vitals["hr"])
    
    def _evolve_cardiac(self):
        """Cardiac event - MI or decompensated HF"""
        # Rising troponin
        if self.elapsed_hours > 1:
            self.vitals["troponin"] += 0.15
        
        # Tachycardia or bradycardia
        if random.random() < 0.7:  # Tachycardia more common
            self.vitals["hr"] += 2.0
        else:
            self.vitals["hr"] -= 1.0
        
        # Blood pressure changes
        if self.elapsed_hours > 3:
            self.vitals["sbp"] -= 1.5  # Hypotension
        
        # Rising lactate
        if self.elapsed_hours > 4:
            self.vitals["lactate"] += 0.2
        
        # Caps
        self.vitals["troponin"] = min(10.0, self.vitals["troponin"])
        self.vitals["hr"] = max(45, min(140, self.vitals["hr"]))
        self.vitals["sbp"] = max(80, self.vitals["sbp"])
    
    def _evolve_falls(self):
        """Fall risk - more subtle changes"""
        # Mild orthostatic changes
        if self.elapsed_hours > 2:
            self.vitals["sbp"] -= 0.5
        
        # This scenario focuses more on functional/cognitive factors
        # which we'll add to payload separately
        pass
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current patient state with realistic values"""
        state = {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "elapsed_hours": round(self.elapsed_hours, 2),
            "scenario": self.scenario,
        }
        
        # Add vitals with noise
        for key, value in self.vitals.items():
            state[key] = round(self._add_noise(value), 2)
        
        # Add scenario-specific features
        if self.scenario == "sepsis":
            state["infection_suspected"] = True
            state["procalcitonin"] = round(0.5 + self.elapsed_hours * 0.3, 2)
            if self.elapsed_hours > 5:
                state["bands"] = random.uniform(12, 20)
        
        if self.scenario == "respiratory":
            state["infiltrates"] = True
            if self.elapsed_hours > 4:
                state["accessory_muscles"] = True
        
        if self.scenario == "cardiac":
            state["chest_pain_severity"] = min(10, 3 + self.elapsed_hours * 0.5)
            state["cad"] = True
            if self.elapsed_hours > 6:
                state["st_elevation"] = True
        
        if self.scenario == "falls":
            state["falls_12m"] = random.randint(1, 3)
            state["unsteady"] = True
            state["med_count"] = random.randint(8, 14)
            state["cognition_issues"] = self.elapsed_hours > 4
        
        return state

# Global patient simulations
active_simulations: Dict[str, PatientSimulator] = {}

def init_simulation_ward():
    """Initialize 10 patients with different scenarios"""
    global active_simulations
    
    scenarios = [
        ("stable", 3),
        ("sepsis", 2),
        ("respiratory", 2),
        ("cardiac", 2),
        ("falls", 1)
    ]
    
    patient_id = 1
    for scenario, count in scenarios:
        for _ in range(count):
            pid = f"PT-{patient_id:03d}"
            active_simulations[pid] = PatientSimulator(pid, scenario)
            patient_id += 1
    # Start one sepsis patient at high risk
    active_simulations["PT-004"].advance_time(6.0)

# Initialize on startup
init_simulation_ward()

# -----------------------------------------------------------------------------
# SIMULATION ROUTES
# -----------------------------------------------------------------------------
@app.get("/simulation/status")
def get_simulation_status():
    """Get current state of all simulated patients"""
    patients = []
    for patient_id, sim in active_simulations.items():
        state = sim.get_current_state()
        patients.append(state)
    
    return {
        "status": "ok",
        "patient_count": len(patients),
        "patients": patients,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/simulation/advance")
def advance_simulation(hours: float = 0.5):
    """Advance simulation time for all patients"""
    for sim in active_simulations.values():
        sim.advance_time(hours)
    
    return {
        "status": "ok",
        "advanced_hours": hours,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/simulation/reset")
def reset_simulation():
    """Reset all simulations"""
    global active_simulations
    active_simulations.clear()
    init_simulation_ward()
    
    return {
        "status": "ok",
        "message": "Simulation reset",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/simulation/monitor")
async def monitor_all_patients():
    """Get real-time predictions for all patients"""
    results = []
    
    for patient_id, sim in active_simulations.items():
        state = sim.get_current_state()
        
        # Run predictions
        sepsis_risk, sepsis_explain, sepsis_time, sepsis_forecast = predict_sepsis(state)
        resp_risk, resp_explain, resp_time, resp_forecast = predict_respiratory(state)
        cardiac_risk, cardiac_explain, cardiac_time, cardiac_forecast = predict_cardiac(state)
        fall_risk, fall_explain, fall_time, fall_forecast = predict_falls(state)
        
        max_risk = max(sepsis_risk, resp_risk, cardiac_risk, fall_risk)
        
        # Determine primary concern
        risks = {
            "sepsis": sepsis_risk,
            "respiratory": resp_risk,
            "cardiac": cardiac_risk,
            "falls": fall_risk
        }
        primary_concern = max(risks, key=risks.get)
        
        results.append({
            "patient_id": patient_id,
            "name": sim.name,
            "age": sim.age,
            "elapsed_hours": round(sim.elapsed_hours, 1),
            "scenario": sim.scenario,
            "max_risk": round(max_risk, 1),
            "primary_concern": primary_concern,
            "overall_severity": "CRITICAL" if max_risk >= 80 else "HIGH" if max_risk >= 60 else "MODERATE" if max_risk >= 40 else "LOW",
            "risks": {
                "sepsis": {
                    "score": round(sepsis_risk, 1),
                    "time_to_event": sepsis_time,
                    "top_factors": sepsis_explain[:3]
                },
                "respiratory": {
                    "score": round(resp_risk, 1),
                    "time_to_event": resp_time,
                    "top_factors": resp_explain[:3]
                },
                "cardiac": {
                    "score": round(cardiac_risk, 1),
                    "time_to_event": cardiac_time,
                    "top_factors": cardiac_explain[:3]
                },
                "falls": {
                    "score": round(fall_risk, 1),
                    "time_to_event": fall_time,
                    "top_factors": fall_explain[:3]
                }
            },
            "vitals": {
                "hr": state.get("hr"),
                "bp": f"{state.get('sbp')}/{state.get('dbp')}",
                "rr": state.get("rr"),
                "spo2": state.get("spo2"),
                "temp": state.get("temperature_c")
            }
        })
    
    # Sort by max risk (highest first)
    results.sort(key=lambda x: x["max_risk"], reverse=True)
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "patient_count": len(results),
        "patients": results
    }

# -----------------------------------------------------------------------------
# WEBSOCKET FOR LIVE UPDATES
# -----------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send updates every 5 seconds
            await asyncio.sleep(5)
            
            # Advance simulation
            for sim in active_simulations.values():
                sim.advance_time(0.5)  # 5 minutes
            
            # Get current state
            patients = []
            for patient_id, sim in active_simulations.items():
                state = sim.get_current_state()
                
                sepsis_risk, _, sepsis_time, sepsis_forecast = predict_sepsis(state)
            resp_risk, _, resp_time, resp_forecast = predict_respiratory(state)
            cardiac_risk, _, cardiac_time, cardiac_forecast = predict_cardiac(state)
            fall_risk, _, fall_time, fall_forecast = predict_falls(state)
                
                max_risk = max(sepsis_risk, resp_risk, cardiac_risk, fall_risk)
                
                patients.append({
                    "patient_id": patient_id,
                    "name": sim.name,
                    "age": sim.age,
                    "elapsed_hours": round(sim.elapsed_hours, 1),
                    "scenario": sim.scenario,
                    "max_risk": round(max_risk, 1),
                    "risks": {
                        "sepsis": round(sepsis_risk, 1),
                        "respiratory": round(resp_risk, 1),
                        "cardiac": round(cardiac_risk, 1),
                        "falls": round(fall_risk, 1)
                    },
                    "vitals": {
                        "hr": round(state.get("hr", 0), 1),
                        "bp": f"{round(state.get('sbp', 0))}/{round(state.get('dbp', 0))}",
                        "rr": round(state.get("rr", 0), 1),
                        "spo2": round(state.get("spo2", 0), 1),
                        "temp": round(state.get("temperature_c", 0), 1)
                    }
                })
            
            await websocket.send_json({
                "type": "update",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "patients": patients
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# -----------------------------------------------------------------------------
# Dashboard Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard at root"""
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Looking for dashboard.html</p>",
            status_code=404
        )

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard"""
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Looking for dashboard.html</p>",
            status_code=404
        )

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "predictwell",
        "version": APP_VERSION,
        "active_patients": len(active_simulations),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# -----------------------------------------------------------------------------
# Local dev entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    uvicorn.run("app_expanded:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)



