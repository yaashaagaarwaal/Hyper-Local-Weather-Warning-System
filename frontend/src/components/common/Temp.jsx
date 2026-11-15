import { useSettings } from "../../context/SettingsContext";
import { formatTemperature } from "../../utils/formatters";

// Renders a Celsius value in the user's preferred unit (Settings page).
export default function Temp({ value, decimals = 1 }) {
  const { tempUnit } = useSettings();
  if (value === undefined || value === null) return null;
  return <>{formatTemperature(value, tempUnit, decimals)}</>;
}
