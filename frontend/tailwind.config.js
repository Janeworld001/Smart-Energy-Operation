/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17221c",
        muted: "#77827a",
        canvas: "#f5f7f5",
        forest: "#1c6248",
        lime: "#c5f06b",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel: "0 2px 12px rgba(27, 44, 34, 0.045)",
      },
    },
  },
  plugins: [],
};
