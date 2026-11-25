// Public home page: cinematic Earth -> India journey, a live weather-desk
// preview, how the system works, features, about and contact. Lives outside
// the application shell (no sidebar); every link goes into the real app.
import "../components/home/home.css";
import HomeNav from "../components/home/HomeNav";
import Journey from "../components/home/Journey";
import WeatherDesk from "../components/home/WeatherDesk";
import HowItWorks from "../components/home/HowItWorks";
import Features from "../components/home/Features";
import { About, Footer } from "../components/home/AboutFooter";

export default function Home() {
  return (
    <div className="home-root">
      <HomeNav />
      <main>
        <Journey />
        <WeatherDesk />
        <HowItWorks />
        <Features />
        <About />
      </main>
      <Footer />
    </div>
  );
}
