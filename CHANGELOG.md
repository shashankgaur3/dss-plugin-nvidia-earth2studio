# Changelog

## Version 0.1.0 - Initial release - 2026-11-24
- Connection preset for a locally deployed FourCastNet NIM, with optional API key header authentication.
- Recipes: run forecast (downloads ERA5 ARCO / GFS initial conditions, then calls the NIM), extract to dataset (points / region, long / wide, derived wind speed and temperature).
- Run-forecast recipe checks NIM health and the input shape before calling the NIM.
- Model registry with FourCastNet NIM 1.1.0 and 2.0.0.
