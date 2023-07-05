import React, { useState } from "react";
import "./predict.css";

// Predict component
const Predict = () => {
  // State values
  const [data, setData] = useState("");
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    Cgpa: "",
  });

  // Input field change handler -> input field herne
  const setFormHandler = (e) => {
    const { name, value } = e.target;
    setFormData((prevFormData) => ({
      ...prevFormData,
      [name]: +value,
    }));
    console.log(formData);
  };

  // Form submitting handler -> form submit garepaxi
  const formSubmitHandler = (e) => {
    e.preventDefault();
    console.log(formData);
    setLoading(true);
    fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // Body baata hamro form ko data pathaako
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        // Status success vayepachi prediction ko response paayeko ra store gareko
        setData(data);
        setLoading(false);
      })
      .catch((error) => {
        // error vaye error
        console.error(error);
      });
  };

  // Clear field button handler

  const clearHandler = () => {
    setFormData({
      Cgpa: "",
    });
    setData(null);
  };

  // Returning JSX
  return (
    <div className="container">
      <p className="title">
        This is react app and is connected to flask in the backend which has ML
        model
      </p>
      <p className="desc">Testing flask react with ML (LinearRegression)</p>
      <form onSubmit={formSubmitHandler}>
        <input
          type="number"
          name="Cgpa"
          placeholder="Enter your Cgpa"
          required="required"
          onChange={setFormHandler}
          value={formData.Cgpa}
        />

        <div className="btn-container">
          <button type="submit">Predict</button>
          <button className="clear" onClick={clearHandler}>
            Clear field
          </button>
        </div>
      </form>
      <div className="data">
        {loading && <h4>Predicting...</h4>}

        {/* Data after making post request in the backend */}
        {data && (
          <>
            <p className="predicted-data">
              The predicted data is <strong>{data?.prediction} lpa </strong>
              <br></br>
              The accuracy is
              <strong>{data?.accuracy} %</strong>
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default Predict;
