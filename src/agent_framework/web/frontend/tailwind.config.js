/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'agent-idle': '#eab308',
        'agent-working': '#22c55e',
        'agent-dead': '#ef4444',
      },
    },
  },
  plugins: [],
}
