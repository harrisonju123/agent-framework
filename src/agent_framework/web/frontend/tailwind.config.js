/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'SF Mono', 'Menlo', 'monospace'],
      },
      colors: {
        'status-idle': '#eab308',
        'status-working': '#22c55e',
        'status-dead': '#ef4444',
        'status-completing': '#3b82f6',
      },
    },
  },
  plugins: [],
}
