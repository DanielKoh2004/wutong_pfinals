# Unified Black Sample Rules

Generated: 2025-12-25 12:01

## Rules Summary

| ID | Name | Weight | Confidence |
|-----|------|--------|------------|
| RULE_01 | High Volume Caller | 35 | 85% |
| RULE_02 | SIM Farm (IMEI Switching) | 25 | 85% |
| RULE_03 | New Account Rapid Activity | 30 | 80% |
| RULE_04 | Device Fingerprint | 20 | 75% |
| RULE_05 | Student Targeting | 20 | 75% |
| RULE_06 | Burst Mode Calling | 30 | 85% |
| RULE_07 | Anonymous Prepaid Burst | 15 | 80% |
| RULE_08 | Multi Device | 15 | 60% |
| RULE_09 | Smishing Pattern | 15 | 65% |
| RULE_10 | Robocall Detection | 30 | 85% |

## Rule Details

### RULE_01: High Volume Caller

- **Description:** High daily call count with dispersion
- **Weight:** 35 points
- **Source:** Task 2: Fraud Portrait + SHAP

### RULE_02: SIM Farm (IMEI Switching)

- **Description:** Multiple IMEI changes (>= 2 times)
- **Weight:** 25 points
- **Source:** Task 2: Fraud Portrait

### RULE_03: New Account Rapid Activity

- **Description:** New account with immediate high activity
- **Weight:** 30 points
- **Source:** Task 2: Fraud Archetypes + SHAP

### RULE_04: Device Fingerprint

- **Description:** High calls on old GSM modems (not 5G)
- **Weight:** 20 points
- **Source:** Task 4: SIM Box Detection

### RULE_05: Student Targeting

- **Description:** Deliberate targeting of student numbers
- **Weight:** 20 points
- **Source:** Task 2: Student Reach

### RULE_06: Burst Mode Calling

- **Description:** High volume (>50 calls) with short duration (<10s avg)
- **Weight:** 30 points
- **Source:** Task 4: Robocall Pattern

### RULE_07: Anonymous Prepaid Burst

- **Description:** Prepaid + unknown channel + high volume
- **Weight:** 15 points
- **Source:** Task 2: Fraud Portrait + SHAP

### RULE_08: Multi Device

- **Description:** Multiple device/IMEI changes
- **Weight:** 15 points
- **Source:** Task 4: Pattern Analysis

### RULE_09: Smishing Pattern

- **Description:** High SMS volume or roaming SMS
- **Weight:** 15 points
- **Source:** Task 4: Pattern Analysis

### RULE_10: Robocall Detection

- **Description:** Extremely high call volume (>=50/day)
- **Weight:** 30 points
- **Source:** Task 2: Robocaller Archetype
