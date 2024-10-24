# gptalter

```terminal
pip install -U git+https://github.com/tnoinkwms/gptalter.git
```

```python
import gptalter

mg = gptalter.MotionGenerator(API =YOUR_API, save = True, show = True)
alter, all_axis, init_value = mg.initalter()
mg.run("say Hello")

#OR

mg.prompt1("pretend to be a ghost")
mg.prompt2()
mg.execute_code()

```
