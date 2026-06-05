/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        zane: {
          primary: '#0ea5e9',
          secondary: '#6366f1',
          accent: '#22c55e',
          dark: '#0f172a',
        },
      },
    },
  },
  plugins: [],
};
