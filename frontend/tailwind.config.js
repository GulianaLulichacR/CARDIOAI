/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          50: "#f1f5fa",
          100: "#dfe8f4",
          200: "#c4d5e9",
          300: "#9bb8d8",
          400: "#6a94c2",
          500: "#4777aa",
          600: "#355e8e",
          700: "#2c4c73",
          800: "#294160",
          900: "#0b2545",
          950: "#071a33"
        },
        accent: {
          50: "#eff8ff",
          100: "#dbeefe",
          200: "#bfe2fe",
          300: "#93d1fd",
          400: "#60b5fa",
          500: "#3b96f6",
          600: "#2577eb",
          700: "#1d60d8",
          800: "#1e4fae",
          900: "#1e448a"
        }
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"]
      },
      boxShadow: {
        card: "0 1px 2px rgba(11,37,69,0.06), 0 6px 24px -12px rgba(11,37,69,0.18)",
        pop: "0 24px 60px -20px rgba(11,37,69,0.35)"
      }
    }
  },
  plugins: []
};
