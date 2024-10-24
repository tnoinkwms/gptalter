# gptalter

```terminal
pip install -U git+https://github.com/tnoinkwms/gptalter.git
```

```python
import gptalter

mg = gptalter.MotionGenerator(API= YOUR_API, gpt3 = "gpt-4o-mini", gpt4 = "gpt-4-turbo", save = False, show=False, serial_device = "/dev/tty.usbserial-FT2KYV79")
alter, all_axis, init_value = mg.initalter()
mg.run("say Hello")

#OR

mg.prompt1("pretend to be a ghost")
mg.prompt2()
mg.execute_code()

```
