# Métodos Numéricos para Determinación de Raíces

Este proyecto implementa y compara tres métodos numéricos para encontrar raíces de ecuaciones: Bisección, Newton-Raphson y Secante.

## Ecuaciones Estudiadas

1. \( x^3 - e^{0.8x} = 20 \)
2. \( 3 \sin(0.5x) - 0.5x + 2 = 0 \)
3. \( x^3 - x^2e^{-0.5x} - 3x = -1 \)
4. \( \cos^2x - 0.5xe^{0.3x} + 5 = 0 \)

## Métodos Implementados

- **Bisección**: Método robusto que requiere un intervalo donde la función cambie de signo
- **Newton-Raphson**: Método rápido que requiere la derivada de la función
- **Secante**: Similar a Newton-Raphson pero sin necesidad de calcular la derivada

## Instalación y Uso

### Requisitos
```bash
pip install numpy matplotlib pandas