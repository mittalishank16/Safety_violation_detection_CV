/** @type {import('next').NextConfig} */
const nextConfig = {
reactStrictMode: true,
// Allows the Next.js app to call your Render API without CORS issues
// during development. In production, CORS is handled by FastAPI middleware.
};
module.exports = nextConfig;