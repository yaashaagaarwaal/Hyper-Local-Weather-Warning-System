import { mockRequest } from "./apiClient";
import { ALERTS, getActiveAlerts, getAlertById, getAlertsCountInRange } from "../data/alertsData";

// GET /alerts
export function fetchAllAlerts() {
  return mockRequest(() => ALERTS);
}

// GET /alerts/active
export function fetchActiveAlerts() {
  return mockRequest(() => getActiveAlerts());
}

// GET /alerts/:id
export function fetchAlertById(id) {
  return mockRequest(() => getAlertById(id));
}

// GET /alerts/count?range=...
export function fetchAlertsCountInRange(rangeKey) {
  return mockRequest(() => getAlertsCountInRange(rangeKey));
}
