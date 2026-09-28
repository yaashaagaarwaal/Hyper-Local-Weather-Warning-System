// Demo locations used across the prototype (Delhi/NCR focus).
// Swap for a geocoding/location API later.
// `savedAs` tags a handful of entries as the user's quick-access
// Home / College / Workplace shortcuts on the Locations page.
export const LOCATIONS = [
  { id: "loc-ndls", name: "New Delhi", region: "Delhi", lat: 28.6139, lng: 77.209, isPrimary: true },
  { id: "loc-sec62", name: "Sector 62", region: "Noida", lat: 28.628, lng: 77.3649, savedAs: "home" },
  { id: "loc-ggn", name: "Cyber Hub", region: "Gurugram", lat: 28.4949, lng: 77.0891, savedAs: "work" },
  { id: "loc-du", name: "North Campus", region: "Delhi University", lat: 28.6874, lng: 77.2098, savedAs: "college" },
  { id: "loc-noida", name: "Noida", region: "Uttar Pradesh", lat: 28.5355, lng: 77.391 },
  { id: "loc-fbd", name: "Faridabad", region: "Haryana", lat: 28.4089, lng: 77.3178 },
  { id: "loc-ghz", name: "Ghaziabad", region: "Uttar Pradesh", lat: 28.6692, lng: 77.4538 },
  { id: "loc-dwk", name: "Dwarka", region: "Delhi", lat: 28.5921, lng: 77.046 },
  { id: "loc-rkp", name: "Rohini", region: "Delhi", lat: 28.7495, lng: 77.0565 },
  { id: "loc-saket", name: "Saket", region: "Delhi", lat: 28.5245, lng: 77.2066 },
];

export const DEFAULT_LOCATION_ID = "loc-ndls";

export const SAVED_LOCATION_META = {
  home: { label: "Home", icon: "home" },
  work: { label: "Workplace", icon: "briefcase" },
  college: { label: "College", icon: "graduation-cap" },
};

export function getSavedLocations() {
  return LOCATIONS.filter((l) => l.savedAs);
}
