// route1.js
const express = require("express");
const router = express.Router();
const record = require("./Model1");

router.use((req, res, next) => {
  console.log('Middleware executed');
  next();
});

router.get('/', (req, res) => {
  res.send('Hello from routerModule');
});

// Endpoint to check user existence and credentials
router.post('/', async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await record.findOne({ email });
    if (user && user.password === password) {
      res.json('exist');
    } else {
      res.json('notexist');
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Internal Server Error" });
  }
});

// Create new user
router.post('/create', async (req, res) => {
  const { name, password, email } = req.body;

  const newUser = new record({
    name,
    password,
    email,
  });

  try {
    const savedRecord = await newUser.save();
    res.json(savedRecord);
  } catch (err) {
    console.error(err);
    res.status(500).send("Internal Server Error");
  }
});

router.post("/Signup", async (req, res) => {
  const { name, password, email } = req.body;

  const data = {
    name,
    password,
    email,
  };

  try {
    const check = await record.findOne({ email });
    if (check) {
      res.json('exist');
    } else {
      await record.insertMany([data]);
      res.json("notexist");
    }
  } catch (e) {
    console.error(e);
    res.json("notexist");
  }
});

module.exports = router;
