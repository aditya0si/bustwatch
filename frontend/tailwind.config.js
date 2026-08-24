/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        meteor: {
          dark: '#0B0F19',
          card: '#131B2E',
          border: '#1E293B',
          accent: '#38BDF8',
        }
      }
    },
  },
  plugins: [],
}
