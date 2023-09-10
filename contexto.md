# Metodología de trabajo en Machine Learning: Contexto

Un E-commerce muy importante, Awamezon de los más grandes en Europa tiene la necesidad de clasificar sus productos en varias categorías, 22 en total.

En la actualidad este proceso se realiza de forma manual, el equipo de producto encontró que este paso corresponde a un 45% del tiempo total desde que se crea el producto, hasta que se publica. Con lo cual, queremos explorar una alternativa que pueda bajar esos tiempos sin incrementar así los costos totales (*1). Por consiguiente, vamos a tener restricciones de latencias que debemos cumplir para asegurarnos de alcanzar la meta.

Este desarrollo se involucran varias áreas: Producto, Marketing, Data science y Engineering. El primer paso es entender a grandes rasgos el objetivo y alinear las expectativas teniendo en cuenta la perspectiva de cada área sobre la oportunidad que se presenta.

Para que se den una idea las categorías son:
```
["All Electronics",
 "Amazon Fashion",
 "Amazon Home",
 "Arts, Crafts & Sewing",
 "Automotive",
 "Books",
 "Camera & Photo",
 "Cell Phones & Accessories",
 "Computers",
 "Digital Music",
 "Grocery",
 "Health & Personal Care",
 "Home Audio & Theater",
 "Industrial & Scientific",
 "Movies & TV",
 "Musical Instruments",
 "Office Products",
 "Pet Supplies",
 "Sports & Outdoors",
 "Tools & Home Improvement",
 "Toys & Games",
 "Video Games"]
```

(*1) El costo está asociado al error de clasificar mal un producto, no es el mismo costo clasificar un producto de "Books" como "Movies & TV" o como "Pet Supplies".

Solo podemos asumir un costo indirecto para este estimado, en el caso de clasificar mal un item incurrimos en costos de revisión y si nunca se detecta puede provocar una pérdida de la conversión.

Además del costo por errores, tenemos el costo por revisión manual. Hoy día todos los productos se revisan de forma manual con excepciones. Entonces, cada vez que un item es correctamente clasificado se descuenta ese costo por revisión manual.
