import { createContext, useContext, useMemo, useState } from "react";
import { LOCATIONS, DEFAULT_LOCATION_ID } from "../data/locations";

const LocationContext = createContext(null);

export function LocationProvider({ children }) {
  const [selectedLocationId, setSelectedLocationId] = useState(DEFAULT_LOCATION_ID);

  const value = useMemo(() => {
    const selectedLocation =
      LOCATIONS.find((l) => l.id === selectedLocationId) ?? LOCATIONS[0];
    return { selectedLocation, selectedLocationId, setSelectedLocationId, locations: LOCATIONS };
  }, [selectedLocationId]);

  return <LocationContext.Provider value={value}>{children}</LocationContext.Provider>;
}

export function useLocationContext() {
  const ctx = useContext(LocationContext);
  if (!ctx) throw new Error("useLocationContext must be used within a LocationProvider");
  return ctx;
}
