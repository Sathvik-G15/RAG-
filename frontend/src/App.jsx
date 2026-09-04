import React from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import PatientPortal from "./pages/PatientPortal";
import DoctorDashboard from "./pages/DoctorDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import CorpusStats from "./pages/CorpusStats";

const navClass = ({ isActive }) => (isActive ? "nav-item active" : "nav-item");

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">CAAR-CDSS</span>
        <nav>
          <NavLink className={navClass} to="/">Patient</NavLink>
          <NavLink className={navClass} to="/doctor">Doctor</NavLink>
          <NavLink className={navClass} to="/admin">Admin</NavLink>
          <NavLink className={navClass} to="/corpus">Corpus</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<PatientPortal />} />
          <Route path="/doctor" element={<DoctorDashboard />} />
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/corpus" element={<CorpusStats />} />
        </Routes>
      </main>
      <footer className="footer">
        Research prototype for academic use. Not a medical device.
      </footer>
    </div>
  );
}
