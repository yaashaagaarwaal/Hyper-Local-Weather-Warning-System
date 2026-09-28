import { createContext, useContext, useMemo, useState } from "react";

const GridFocusContext = createContext(null);

// Cross-page navigation glue: lets a grid clicked on one page (map popup,
// alert card) drive what the next page shows, without prop-drilling through
// the router. Also hosts the scripted "severe weather" demo scenario used to
// give a clean, repeatable talking point when presenting the prototype.
export function GridFocusProvider({ children }) {
  const [focusedGridId, setFocusedGridId] = useState(null);
  const [severeScenario, setSevereScenario] = useState(null);

  const value = useMemo(
    () => ({
      focusedGridId,
      setFocusedGridId,
      clearFocusedGrid: () => setFocusedGridId(null),
      severeScenario,
      triggerSevereScenario: (cell) =>
        setSevereScenario({
          gridId: cell.id,
          hazardType: cell.hazardType,
          rainfallProbability: 87,
          floodRisk: 81,
          overallRisk: 84,
        }),
      dismissSevereScenario: () => setSevereScenario(null),
    }),
    [focusedGridId, severeScenario]
  );

  return <GridFocusContext.Provider value={value}>{children}</GridFocusContext.Provider>;
}

export function useGridFocus() {
  const ctx = useContext(GridFocusContext);
  if (!ctx) throw new Error("useGridFocus must be used within a GridFocusProvider");
  return ctx;
}
