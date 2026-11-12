import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./layout/Layout";
import Home from "./pages/Home";
import AddSign from "./pages/AddSign";
import LivePredict from "./pages/LivePredict";
import Login from "./pages/Login";
import Register from "./pages/Register";
import MovementRecognition from "./pages/MovementRecognition";
import "./App.css";
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="add-sign" element={<AddSign />} />
          <Route path="live" element={<LivePredict />} />
          <Route path="login" element={<Login />} />
          <Route path="register" element={<Register />} />
          <Route path="/movement" element={<MovementRecognition />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
