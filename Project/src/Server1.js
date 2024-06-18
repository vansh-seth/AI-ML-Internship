const express = require('express');
const app = express();
const cors = require('cors');
const mongoose = require('mongoose');
const router = require('./route1');

app.use(cors());
app.use(express.json());

mongoose.connect("mongodb+srv://vansh:vanshseth%402004@auth.pyxmqul.mongodb.net/myDatabaseName?retryWrites=true&w=majority")
  .then(() => {
    console.log("MongoDB connected successfully");
    app.use("/", router);
    app.listen(3001, function() {
      console.log("Express server is running on port 3001");
    });
  })
  .catch(err => {
    console.error("MongoDB connection error:", err);
  });
