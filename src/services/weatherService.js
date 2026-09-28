import { mockRequest } from "./apiClient";
import {
  getCurrentWeather,
  getHourlyTrend,
  getWeeklyOutlook,
} from "../data/weatherData";
import { LOCATIONS } from "../data/locations";

// GET /weather/current?locationId=...
export function fetchCurrentWeather(locationId) {
  return mockRequest(() => getCurrentWeather(locationId));
}

// GET /weather/hourly?locationId=...
export function fetchHourlyTrend(locationId) {
  return mockRequest(() => getHourlyTrend(locationId));
}

// GET /weather/weekly?locationId=...
export function fetchWeeklyOutlook(locationId) {
  return mockRequest(() => getWeeklyOutlook(locationId));
}

// GET /locations
export function fetchLocations() {
  return mockRequest(() => LOCATIONS);
}
