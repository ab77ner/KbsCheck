import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

LABELS = [
    "Normal",
    "InitialAccess",
    "PrivilegeEscalation",
    "Persistence",
    "LateralMovement",
    "ContainerEscape",
    "DataExfiltration",
    "Misconfiguration",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

VERB_RISK = {
    "get": 0.05, "list": 0.10, "watch": 0.10,
    "create": 0.30, "update": 0.35, "patch": 0.40,
    "delete": 0.70, "deletecollection": 0.85,
    "connect": 0.75, "impersonate": 0.95,
}
SENSITIVE_RESOURCES = {
    "secrets": 1.0,
    "configmaps": 0.55,
    "serviceaccounts": 0.75,
    "roles": 0.80,
    "rolebindings": 0.85,
    "clusterroles": 0.90,
    "clusterrolebindings": 0.95,
    "nodes": 0.80,
    "pods": 0.25,
    "pods/exec": 0.95,
    "deployments": 0.35,
    "namespaces": 0.60,
    "certificatesigningrequests": 0.85,
}
ROLE_LEVEL = {
    "cluster-admin": 1.0,
    "admin": 0.85,
    "developer": 0.45,
    "dev": 0.45,
    "edit": 0.55,
    "viewer": 0.15,
    "view": 0.20,
    "readonly": 0.15,
    "anonymous": 0.05,
    "unknown": 0.10,
}
WRITE_VERBS = {"create", "update", "patch", "delete", "deletecollection", "connect", "impersonate"}


def stable_hash(value: Any, modulo: int = 1000) -> int:
    s = str(value)
    h = hashlib.md5(s.encode('utf-8', errors='ignore')).hexdigest()
    return int(h[:8], 16) % modulo


def parse_timestamp(ts: Any) -> float:
    if not ts:
        return 0.0
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        s = str(ts).replace('Z', '+00:00')
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def normalize_status(code: Any) -> float:
    try: c = int(code)
    except Exception: c = 0
    if c <= 0: return 0.0
    return min(c / 600.0, 1.0)


def infer_role(user: str, groups: List[str] = None, explicit_role: str = None) -> str:
    if explicit_role:
        r = str(explicit_role).lower()
        if r in ROLE_LEVEL:
            return r
        if 'cluster-admin' in r or 'masters' in r: return 'cluster-admin'
        if 'admin' in r: return 'admin'
        if 'dev' in r or 'developer' in r or 'edit' in r: return 'developer'
        if 'view' in r or 'readonly' in r: return 'viewer'
    groups = groups or []
    joined = ' '.join([str(user)] + [str(g) for g in groups]).lower()
    if 'system:masters' in joined or 'cluster-admin' in joined: return 'cluster-admin'
    if 'admin' in joined: return 'admin'
    if 'edit' in joined or 'developer' in joined or 'dev' in joined: return 'developer'
    if 'view' in joined or 'readonly' in joined or 'viewer' in joined: return 'viewer'
    if 'anonymous' in joined: return 'anonymous'
    return 'unknown'


def event_to_feature(event: Dict[str, Any], prev_event: Optional[Dict[str, Any]] = None) -> List[float]:
    user = event.get('user', 'unknown')
    role = infer_role(user, event.get('groups', []), event.get('role'))
    verb = str(event.get('verb', 'unknown')).lower()
    resource = str(event.get('resource', 'unknown')).lower()
    namespace = event.get('namespace', '') or ''
    source_ip = event.get('source_ip', 'unknown')
    code = int(event.get('status_code', event.get('response_code', 0)) or 0)
    ts = event.get('timestamp', None)
    if ts is None:
        ts = parse_timestamp(event.get('ts') or event.get('requestReceivedTimestamp') or event.get('stageTimestamp'))

    delta_t = 0.0
    same_user_prev = 0.0
    same_ip_prev = 0.0
    if prev_event:
        prev_ts = prev_event.get('timestamp', None)
        if prev_ts is None:
            prev_ts = parse_timestamp(prev_event.get('ts') or prev_event.get('requestReceivedTimestamp') or prev_event.get('stageTimestamp'))
        delta_t = max(0.0, min((float(ts or 0.0) - float(prev_ts or 0.0)) / 3600.0, 1.0))
        same_user_prev = 1.0 if user == prev_event.get('user') else 0.0
        same_ip_prev = 1.0 if source_ip == prev_event.get('source_ip') else 0.0

    role_score = float(event.get('role_privilege_score', ROLE_LEVEL.get(role, ROLE_LEVEL['unknown'])))
    if role_score > 1.0: role_score = min(role_score / 10.0, 1.0)
    res_score = float(event.get('resource_privilege_score', SENSITIVE_RESOURCES.get(resource, 0.20)))
    if res_score > 1.0: res_score = min(res_score / 10.0, 1.0)
    verb_score = VERB_RISK.get(verb, 0.20)
    denied = 1.0 if code in (401, 403) else 0.0
    mutating = float(event.get('mutating', 1.0 if verb in WRITE_VERBS else 0.0))
    is_sensitive = float(event.get('is_sensitive', 1.0 if resource in SENSITIVE_RESOURCES and SENSITIVE_RESOURCES[resource] >= 0.75 else 0.0))
    is_cluster_scoped = float(event.get('is_cluster_scoped', 1.0 if namespace == '' else 0.0))
    rbac_allowed = float(event.get('rbac_allowed', 0.0 if denied else 1.0))
    rbac_violation = float(event.get('rbac_violation', 1.0 if denied or (role in {'viewer','readonly','anonymous'} and verb in WRITE_VERBS) else 0.0))
    risk_score = float(event.get('risk_score', min(1.0, 0.35 * res_score + 0.35 * verb_score + 0.20 * denied + 0.10 * rbac_violation)))
    graph_risk = float(event.get('graph_risk', risk_score))
    privilege_gap = float(event.get('privilege_gap', max(0.0, res_score - role_score)))
    role_out_degree = float(event.get('role_out_degree', stable_hash(role, 100) / 100.0))
    resource_in_degree = float(event.get('resource_in_degree', stable_hash(resource, 100) / 100.0))
    hour = float(event.get('hour', 0.0))
    if not hour and ts:
        try: hour = datetime.fromtimestamp(float(ts)).hour / 24.0
        except Exception: hour = 0.0
    elif hour > 1.0:
        hour = hour / 24.0

    # 19 dimensions, including RBAC graph features from previous data when present.
    return [
        stable_hash(user, 1000) / 1000.0,
        stable_hash(role, 100) / 100.0,
        stable_hash(verb, 100) / 100.0,
        stable_hash(resource, 200) / 200.0,
        stable_hash(namespace, 100) / 100.0,
        stable_hash(source_ip, 1000) / 1000.0,
        normalize_status(code),
        hour,
        delta_t,
        same_user_prev,
        same_ip_prev,
        mutating,
        is_sensitive,
        is_cluster_scoped,
        rbac_allowed,
        rbac_violation,
        risk_score,
        graph_risk,
        privilege_gap + 0.5 * role_out_degree + 0.5 * resource_in_degree,
    ]


def parse_audit_event(raw: Any) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        user_info = d.get('user', {}) or {}
        obj = d.get('objectRef', {}) or {}
        status = d.get('responseStatus', {}) or {}
        username = user_info.get('username', d.get('user', 'unknown'))
        groups = user_info.get('groups', []) if isinstance(user_info, dict) else []
        verb = d.get('verb', 'unknown') or 'unknown'
        resource = obj.get('resource', d.get('resource', 'unknown')) or 'unknown'
        subresource = obj.get('subresource', '')
        if subresource:
            resource = f'{resource}/{subresource}'
        namespace = obj.get('namespace', d.get('namespace', '')) or ''
        code = int(status.get('code', d.get('response_code', 0)) or 0)
        source_ips = d.get('sourceIPs', []) or []
        source_ip = source_ips[0] if source_ips else d.get('source_ip', 'unknown')
        role = infer_role(username, groups, d.get('role'))
        return {
            'timestamp': parse_timestamp(d.get('requestReceivedTimestamp') or d.get('stageTimestamp') or d.get('timestamp') or d.get('ts')),
            'ts': d.get('ts'),
            'user': username,
            'groups': groups,
            'role': role,
            'verb': verb,
            'resource': resource,
            'namespace': namespace,
            'status_code': code,
            'response_code': code,
            'source_ip': source_ip,
            'request_uri': d.get('requestURI', ''),
        }
    except Exception:
        return None


def rule_label_event(event: Dict[str, Any]) -> int:
    role = infer_role(event.get('user','unknown'), event.get('groups',[]), event.get('role'))
    verb = str(event.get('verb', 'unknown')).lower()
    resource = str(event.get('resource', 'unknown')).lower()
    code = int(event.get('status_code', event.get('response_code', 0)) or 0)
    if event.get('label') in LABEL_TO_ID:
        return LABEL_TO_ID[event['label']]
    if code in (401, 403) and verb in WRITE_VERBS: return LABEL_TO_ID['PrivilegeEscalation']
    if resource in {'secrets','clusterroles','clusterrolebindings','nodes','pods/exec'} and role in {'viewer','readonly','anonymous','unknown'}: return LABEL_TO_ID['PrivilegeEscalation']
    if resource in {'secrets','configmaps','serviceaccounts'}: return LABEL_TO_ID['DataExfiltration']
    if verb in {'list','watch'} and resource in {'pods','namespaces','nodes','services','deployments'}: return LABEL_TO_ID['InitialAccess']
    return LABEL_TO_ID['Normal']


def sample_to_features(sample: Dict[str, Any], seq_len: int = 20) -> Dict[str, Any]:
    events = sample.get('events') or []
    label_name = sample.get('label', 'Normal')
    label = LABEL_TO_ID.get(label_name, LABEL_TO_ID['Normal'])
    feats=[]; prev=None
    for ev in events:
        feats.append(event_to_feature(ev, prev)); prev=ev
    return {
        'sample_id': sample.get('sample_id'),
        'label': label,
        'label_name': label_name,
        'chain_id': sample.get('chain_id'),
        'chain_name': sample.get('chain_name'),
        'features': feats[:seq_len],
        'events_count': len(events),
    }


def build_sequences(events: List[Dict[str, Any]], seq_len: int = 20, stride: int = 1, default_label: Optional[int] = None):
    feats=[]; prev=None
    for ev in events:
        feats.append(event_to_feature(ev, prev)); prev=ev
    rows=[]
    for i in range(0, max(0, len(feats)-seq_len+1), stride):
        ev_seq=events[i:i+seq_len]
        if default_label is not None:
            label=int(default_label)
        else:
            labels=[rule_label_event(e) for e in ev_seq]
            label=max(labels) if labels else 0
        rows.append({'features': feats[i:i+seq_len], 'label': label})
    return rows
