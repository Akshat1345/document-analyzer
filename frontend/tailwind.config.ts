import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        slateBg: "#0f172a",
        cardBg: "#1e293b",
      },
    },
  },
  plugins: [],
};

export default config;
