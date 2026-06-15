ATTACK_STAGE_MAP = {
    0: 'Normal',
    1: 'Initial Access',
    2: 'Privilege Escalation',
    3: 'Persistence',
    4: 'Lateral Movement',
    5: 'Container Escape',
    6: 'Data Exfiltration',
    7: 'Misconfiguration',
}


def reconstruct_attack_chain(events, labels):
    stages = []
    for idx, label in enumerate(labels):
        label = int(label)
        stage = ATTACK_STAGE_MAP.get(label, 'Unknown')
        if label != 0:
            stages.append({'index': idx, 'label': label, 'stage': stage, 'event': events[idx] if idx < len(events) else None})
    risk = min(1.0, len(stages) / max(1, len(labels)) + 0.12 * len(set(x['label'] for x in stages)))
    return {'stages': stages, 'risk_score': risk, 'summary': ' -> '.join(x['stage'] for x in stages) if stages else 'Normal'}
