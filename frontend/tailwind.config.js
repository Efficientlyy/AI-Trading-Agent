/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          light: '#4da3ff',
          DEFAULT: '#0078ff',
          dark: '#0057cc',
        },
        secondary: {
          light: '#6c757d',
          DEFAULT: '#495057',
          dark: '#343a40',
        },
        success: {
          light: '#28a745',
          DEFAULT: '#198754',
          dark: '#0f5132',
        },
        danger: {
          light: '#dc3545',
          DEFAULT: '#dc3545',
          dark: '#842029',
        },
        warning: {
          light: '#ffc107',
          DEFAULT: '#ffc107',
          dark: '#664d03',
        },
        info: {
          light: '#0dcaf0',
          DEFAULT: '#0dcaf0',
          dark: '#055160',
        },
      },
    },
  },
  plugins: [],
}
