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

# OR

mg = gptalter.MotionGenerator(API= YOUR_API, OSC =True, ip = "127.0.0.1", port = 60001)
send_osc_data, all_axis, init_value = mg.initalter()
send_osc_data([1,2],[0,255])
mg.run("say Hello")

```
